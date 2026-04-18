from transformers import AutoTokenizer
import torch
import json
def process_qa_sample(
    question: str,
    context: str,
    answer: dict = None,
    tokenizer=None,
    max_length: int = 512,
    doc_stride: int = 128,
):
    """
    Tokenize một QA sample và tìm vị trí token của answer span.

    Args:
        question    : Câu hỏi (str).
        context     : Đoạn văn chứa câu trả lời (str).
        answer      : {'text': [...], 'answer_start': [...]} nếu là SQuAD.
                      None hoặc {'text': [], 'answer_start': []} nếu là ViQuAD (unanswerable).
        tokenizer   : HuggingFace fast tokenizer (BẮT BUỘC use_fast=True).
        max_length  : Độ dài tối đa của chuỗi token.
        doc_stride  : Bước trượt khi context bị cắt.

    Returns:
        input_ids        : LongTensor [max_length]
        attention_mask   : LongTensor [max_length]
        start_position   : LongTensor scalar — token index của đầu answer (0 nếu unanswerable)
        end_position     : LongTensor scalar — token index của cuối answer (0 nếu unanswerable)
    """

    # ------------------------------------------------------------------
    # TODO 2.1 — Tokenize question + context
    # ------------------------------------------------------------------
    inputs = tokenizer(
        question,
        context,
        max_length=max_length,
        truncation="only_second",
        stride=doc_stride,
        padding="max_length",
        return_overflowing_tokens=False,
        return_offsets_mapping=True,
        return_tensors="pt",
    )

    input_ids      = inputs["input_ids"].squeeze(0)      # [max_length]
    attention_mask = inputs["attention_mask"].squeeze(0)   # [max_length]
    offset_mapping = inputs["offset_mapping"].squeeze(0)   # [max_length, 2]

    # Mặc định gán bằng 0 (unanswerable)
    start_position = torch.tensor(0, dtype=torch.long)
    end_position   = torch.tensor(0, dtype=torch.long)

    # ------------------------------------------------------------------
    # TODO 2.2 — Xác định vùng context trong chuỗi token
    # ------------------------------------------------------------------
    sequence_ids = inputs.sequence_ids(0)
    
    # Tìm index token đầu tiên của context (sequence_id == 1)
    token_start_index = 0
    while token_start_index < len(sequence_ids) and sequence_ids[token_start_index] != 1:
        token_start_index += 1

    # Tìm index token cuối cùng của context
    token_end_index = len(sequence_ids) - 1
    while token_end_index >= 0 and sequence_ids[token_end_index] != 1:
        token_end_index -= 1

    # ------------------------------------------------------------------
    # TODO 2.3 & 2.4 — Map character offset → token index & Xử lý edge cases
    # ------------------------------------------------------------------
    if answer is not None and "answer_start" in answer and len(answer["answer_start"]) > 0:
        start_char = answer["answer_start"][0]
        end_char   = start_char + len(answer["text"][0])

        # Kiểm tra xem answer span có nằm ngoài phần context còn lại sau khi truncate không
        if start_char < offset_mapping[token_start_index][0] or end_char > offset_mapping[token_end_index][1]:
            # Nằm ngoài vùng -> giữ nguyên start_position = 0, end_position = 0
            pass 
        else:
            # Tìm token bắt đầu answer
            idx = token_start_index
            while idx <= token_end_index and offset_mapping[idx][0] <= start_char:
                idx += 1
            start_position = torch.tensor(idx - 1, dtype=torch.long)

            # Tìm token kết thúc answer
            idx = token_end_index
            while idx >= token_start_index and offset_mapping[idx][1] >= end_char:
                idx -= 1
            end_position = torch.tensor(idx + 1, dtype=torch.long)

    # Xóa offset_mapping khỏi kết quả trả về để giống với format gốc
    return input_ids, attention_mask, start_position, end_position
import json

def load_squad_data(file_path):
    """Đọc và bóc tách dữ liệu SQuAD format."""
    
    # 1. Đọc file JSON
    with open(file_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    parsed_data = []

    # 2. Duyệt qua các tầng của file JSON
    for article in dataset['data']:
        # Lặp qua từng đoạn văn (paragraph) trong bài viết
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            
            # Lặp qua từng câu hỏi (qas) trong đoạn văn
            for qa in paragraph['qas']:
                question_id = qa['id']
                question = qa['question']
                
                # SQuAD dev set thường có nhiều câu trả lời cho 1 câu hỏi.
                # Khi tokenize, ta thường chỉ lấy câu trả lời đầu tiên làm mốc.
                answer_dict = None
                if 'answers' in qa and len(qa['answers']) > 0:
                    first_answer = qa['answers'][0]
                    answer_dict = {
                        "text": [first_answer['text']],
                        "answer_start": [first_answer['answer_start']]
                    }
                else:
                    # Xử lý trường hợp unanswerable (như SQuAD 2.0 hoặc ViQuAD)
                    answer_dict = {"text": [], "answer_start": []}
                
                # 3. Đưa vào danh sách dưới dạng dictionary phẳng
                parsed_data.append({
                    "id": question_id,
                    "question": question,
                    "context": context,
                    "answer": answer_dict
                })
                
    return parsed_data
# ============================================================
# TODO 2.5 — Unit test (chạy thẳng file này để kiểm tra)
# ============================================================
# if __name__ == "__main__":
#     tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base", use_fast=True)

#     # --- Test 1: SQuAD — có answer rõ ràng ---
#     ids, mask, s, e = process_qa_sample(
#         question="When did the first World Cup happen?",
#         context="The first World Cup was held in 1930 in Uruguay.",
#         answer={"text": ["1930"], "answer_start": [28]},
#         tokenizer=tokenizer,
#     )
#     print(f"[Test 1 - SQuAD]  start={s.item()}, end={e.item()}")
#     # Kỳ vọng: s > 0, e >= s
#     # Verify bằng cách decode lại token:
#     print("  Answer tokens:", tokenizer.decode(ids[s : e + 1]))
#     # Kỳ vọng output: "1930"

#     # --- Test 2: ViQuAD — unanswerable (answer=None) ---
#     ids, mask, s, e = process_qa_sample(
#         question="Ai là tổng thống Việt Nam?",
#         context="Việt Nam có thủ đô là Hà Nội.",
#         answer=None,
#         tokenizer=tokenizer,
#     )
#     print(f"[Test 2 - VI unanswerable]  start={s.item()}, end={e.item()}")
#     # Kỳ vọng: s == 0, e == 0

#     # --- Test 3: SQuAD — answer bị truncate (nằm ngoài max_length nhỏ) ---
#     ids, mask, s, e = process_qa_sample(
#         question="What is at the end?",
#         context="Word " * 200 + "ANSWER is here.",
#         answer={"text": ["ANSWER"], "answer_start": 1000},
#         tokenizer=tokenizer,
#         max_length=64,
#     )
#     print(f"[Test 3 - Truncated]  start={s.item()}, end={e.item()}")
    # Kỳ vọng: s == 0, e == 0  (answer bị cắt mất)
# ==========================================
# TEST VỚI CÁC BỘ DATASET
# ==========================================
if __name__ == "__main__":
    import json
    import os
    from tqdm import tqdm
    from transformers import AutoTokenizer

    # Thư mục chứa dữ liệu của bạn
    base_dir = r"D:\process_qa_sample\test\ViQuAD v2.0"
    
    # Danh sách các file cần chạy
    files_to_process = [
        "dev (public_test).json",
        "test (private_test).json",
        "train.json"
    ]

    # Khởi tạo Tokenizer một lần duy nhất bên ngoài vòng lặp để tiết kiệm thời gian
    print("Đang tải Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base", use_fast=True)

    # Lặp qua từng file để xử lý
    for file_name in files_to_process:
        print(f"\n{'='*60}")
        print(f"BẮT ĐẦU XỬ LÝ FILE: {file_name}")
        print(f"{'='*60}")

        # Tạo đường dẫn đầy đủ tới file input
        input_path = os.path.join(base_dir, file_name)
        
        # Tạo tên file output (ví dụ: "dev-v1.1_output.jsonl")
        # os.path.splitext sẽ tách "dev-v1.1" và ".json"
        base_name = os.path.splitext(file_name)[0] 
        output_name = f"{base_name}_output.jsonl" 
        output_path = os.path.join(base_dir, output_name)

        # 1. Đọc dữ liệu
        print(f"Đang đọc dữ liệu từ {file_name}...")
        parsed_dataset = load_squad_data(input_path)
        print(f"Đã đọc thành công {len(parsed_dataset)} câu hỏi.")

        # 2. Tokenize dữ liệu
        final_training_data = []
        # desc giúp hiển thị tên file trên thanh tiến trình cho dễ theo dõi
        for sample in tqdm(parsed_dataset, desc=f"Tokenizing {file_name}"):
            ids, mask, s, e = process_qa_sample(
                question=sample['question'],
                context=sample['context'],
                answer=sample['answer'],
                tokenizer=tokenizer,
                max_length=512,
                doc_stride=128
            )
            
            final_training_data.append({
                "id": sample["id"],
                "input_ids": ids.tolist(),
                "attention_mask": mask.tolist(),
                "start_positions": s.item(),
                "end_positions": e.item()
            })

        # 3. Lưu kết quả ra file
        print(f"Đang lưu kết quả ra file: {output_name}...")
        with open(output_path, "w", encoding="utf-8") as f:
            for item in final_training_data:
                f.write(json.dumps(item) + "\n")

        print(f"HOÀN TẤT FILE: {file_name}!\n")

    print("🎉 ĐÃ XỬ LÝ THÀNH CÔNG TOÀN BỘ 4 FILE!")