# process_qa_sample.py
import json
import torch
from transformers import AutoTokenizer


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
        answer      : {'text': [...], 'answer_start': [...]} — answer_start PHẢI là list[int].
                      None hoặc {'text': [], 'answer_start': []} nếu unanswerable.
        tokenizer   : HuggingFace fast tokenizer (BẮT BUỘC use_fast=True).
        max_length  : Độ dài tối đa của chuỗi token.
        doc_stride  : Bước trượt khi context bị cắt.

    Returns:
        input_ids        : LongTensor [max_length]
        attention_mask   : LongTensor [max_length]
        start_position   : LongTensor scalar — token index của đầu answer (0 nếu unanswerable)
        end_position     : LongTensor scalar — token index của cuối answer (0 nếu unanswerable)
        question_end     : LongTensor scalar — token index của [SEP] đầu tiên (= cuối question)
    """

    # ------------------------------------------------------------------
    # 2.1 — Tokenize question + context
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

    input_ids      = inputs["input_ids"].squeeze(0)       # [max_length]
    attention_mask = inputs["attention_mask"].squeeze(0)  # [max_length]
    offset_mapping = inputs["offset_mapping"].squeeze(0)  # [max_length, 2]

    # Mặc định: unanswerable
    start_position = torch.tensor(0, dtype=torch.long)
    end_position   = torch.tensor(0, dtype=torch.long)

    # ------------------------------------------------------------------
    # 2.2 — Tìm question_end: index của [SEP] đầu tiên
    # Model_core dùng để xác định question tokens khi subsampling.
    # ------------------------------------------------------------------
    sep_token_id = tokenizer.sep_token_id
    sep_positions = (input_ids == sep_token_id).nonzero(as_tuple=True)[0]
    # sep_positions[0] = [SEP] sau question, sep_positions[1] = [SEP] sau context
    question_end = sep_positions[0] if len(sep_positions) > 0 else torch.tensor(0)
    question_end = question_end.to(torch.long)

    # ------------------------------------------------------------------
    # 2.3 — Xác định vùng context trong chuỗi token (sequence_id == 1)
    # ------------------------------------------------------------------
    sequence_ids = inputs.sequence_ids(0)

    token_start_index = 0
    while token_start_index < len(sequence_ids) and sequence_ids[token_start_index] != 1:
        token_start_index += 1

    token_end_index = len(sequence_ids) - 1
    while token_end_index >= 0 and sequence_ids[token_end_index] != 1:
        token_end_index -= 1

    # ------------------------------------------------------------------
    # 2.4 — Map character offset → token index
    # FIX: Sửa lại logic tìm start/end, tránh off-by-one bug của code gốc.
    # FIX: answer_start phải là list[int] — không phải int thuần.
    # ------------------------------------------------------------------
    is_answerable = (
        answer is not None
        and isinstance(answer.get("answer_start"), list)
        and len(answer["answer_start"]) > 0
        and len(answer.get("text", [])) > 0
    )

    if is_answerable:
        start_char = int(answer["answer_start"][0])
        end_char   = start_char + len(answer["text"][0])

        # Kiểm tra answer có bị truncate không
        context_start_offset = offset_mapping[token_start_index][0].item()
        context_end_offset   = offset_mapping[token_end_index][1].item()

        if start_char < context_start_offset or end_char > context_end_offset:
            # Answer bị cắt → giữ nguyên (0, 0) = unanswerable
            pass
        else:
            # Tìm start token: token cuối cùng có offset_start <= start_char
            # (duyệt từ trái sang, dừng khi vượt quá start_char)
            s_idx = token_start_index
            for idx in range(token_start_index, token_end_index + 1):
                if offset_mapping[idx][0].item() <= start_char:
                    s_idx = idx
                else:
                    break

            # Tìm end token: token đầu tiên có offset_end >= end_char
            # (duyệt từ phải sang, dừng khi nhỏ hơn end_char)
            e_idx = token_end_index
            for idx in range(token_end_index, token_start_index - 1, -1):
                if offset_mapping[idx][1].item() >= end_char:
                    e_idx = idx
                else:
                    break

            # Sanity check: start không được vượt end
            if s_idx <= e_idx:
                start_position = torch.tensor(s_idx, dtype=torch.long)
                end_position   = torch.tensor(e_idx, dtype=torch.long)

    return input_ids, attention_mask, start_position, end_position, question_end


# ------------------------------------------------------------------
# Hàm đọc file SQuAD-format JSON local
# ------------------------------------------------------------------
def load_squad_data(file_path: str) -> list[dict]:
    """Đọc và bóc tách dữ liệu SQuAD format từ file JSON local."""
    with open(file_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    parsed_data = []
    for article in dataset["data"]:
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                # FIX: answer_start luôn là list[int]
                if qa.get("answers") and len(qa["answers"]) > 0:
                    first = qa["answers"][0]
                    answer_dict = {
                        "text": [first["text"]],
                        "answer_start": [int(first["answer_start"])],  # ← list[int]
                    }
                else:
                    answer_dict = {"text": [], "answer_start": []}

                parsed_data.append({
                    "id": qa["id"],
                    "question": qa["question"],
                    "context": context,
                    "answer": answer_dict,
                })

    return parsed_data


# ============================================================
# Unit Test — chạy: python process_qa_sample.py
# ============================================================
if __name__ == "__main__":
    import os
    from tqdm import tqdm

    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base", use_fast=True)

    # --- Test 1: SQuAD — answer rõ ràng ---
    ids, mask, s, e, qend = process_qa_sample(
        question="When did the first World Cup happen?",
        context="The first World Cup was held in 1930 in Uruguay.",
        answer={"text": ["1930"], "answer_start": [28]},  # ← list[int]
        tokenizer=tokenizer,
    )
    decoded = tokenizer.decode(ids[s: e + 1])
    print(f"[Test 1 - SQuAD]  start={s.item()}, end={e.item()}, question_end={qend.item()}")
    print(f"  Answer tokens: '{decoded}'")
    assert s.item() > 0 and e.item() >= s.item(), "Test 1 FAILED: s/e không hợp lệ"
    assert "1930" in decoded, f"Test 1 FAILED: decode ra '{decoded}', kỳ vọng '1930'"
    print("  ✓ Test 1 PASSED\n")

    # --- Test 2: Unanswerable (answer=None) ---
    ids, mask, s, e, qend = process_qa_sample(
        question="Ai là tổng thống Việt Nam?",
        context="Việt Nam có thủ đô là Hà Nội.",
        answer=None,
        tokenizer=tokenizer,
    )
    print(f"[Test 2 - Unanswerable]  start={s.item()}, end={e.item()}, question_end={qend.item()}")
    assert s.item() == 0 and e.item() == 0, "Test 2 FAILED: kỳ vọng (0, 0)"
    print("  ✓ Test 2 PASSED\n")

    # --- Test 3: Answer bị truncate (nằm ngoài max_length nhỏ) ---
    ids, mask, s, e, qend = process_qa_sample(
        question="What is at the end?",
        context="Word " * 200 + "ANSWER is here.",
        answer={"text": ["ANSWER"], "answer_start": [1000]},  # ← FIX: list[int], không phải int
        tokenizer=tokenizer,
        max_length=64,
    )
    print(f"[Test 3 - Truncated]  start={s.item()}, end={e.item()}, question_end={qend.item()}")
    assert s.item() == 0 and e.item() == 0, "Test 3 FAILED: kỳ vọng (0, 0) vì bị truncate"
    print("  ✓ Test 3 PASSED\n")

    # --- Test 4: question_end hợp lệ ---
    ids, mask, s, e, qend = process_qa_sample(
        question="What color is the sky?",
        context="The sky is blue on a clear day.",
        answer={"text": ["blue"], "answer_start": [11]},
        tokenizer=tokenizer,
    )
    print(f"[Test 4 - question_end]  question_end={qend.item()}")
    assert ids[qend.item()].item() == tokenizer.sep_token_id, \
        "Test 4 FAILED: question_end không trỏ vào [SEP]"
    print("  ✓ Test 4 PASSED\n")

    print("✅ Tất cả unit test PASSED!")

    # --- Batch processing trên file local (nếu có) ---
    base_dir = r"D:\process_qa_sample\test\ViQuAD v2.0"
    files_to_process = [
        "dev (public_test).json",
        "test (private_test).json",
        "train.json",
    ]

    for file_name in files_to_process:
        input_path = os.path.join(base_dir, file_name)
        if not os.path.exists(input_path):
            print(f"[SKIP] Không tìm thấy file: {input_path}")
            continue

        print(f"\n{'='*60}")
        print(f"XỬ LÝ: {file_name}")
        print(f"{'='*60}")

        parsed_dataset = load_squad_data(input_path)
        print(f"Đã đọc {len(parsed_dataset):,} câu hỏi.")

        base_name   = os.path.splitext(file_name)[0]
        output_path = os.path.join(base_dir, f"{base_name}_output.jsonl")

        final_data = []
        for sample in tqdm(parsed_dataset, desc=f"Tokenizing {file_name}"):
            ids, mask, s, e, qend = process_qa_sample(
                question=sample["question"],
                context=sample["context"],
                answer=sample["answer"],
                tokenizer=tokenizer,
                max_length=512,
                doc_stride=128,
            )
            final_data.append({
                "id":               sample["id"],
                "input_ids":        ids.tolist(),
                "attention_mask":   mask.tolist(),
                "start_positions":  s.item(),
                "end_positions":    e.item(),
                "question_end":     qend.item(),   # ← thêm mới
            })

        with open(output_path, "w", encoding="utf-8") as f:
            for item in final_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        print(f"Đã lưu: {output_path}")

    print("\n🎉 Hoàn tất!")