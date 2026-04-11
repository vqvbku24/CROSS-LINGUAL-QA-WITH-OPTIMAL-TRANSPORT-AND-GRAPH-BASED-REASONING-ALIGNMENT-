# process_qa_sample.py
###
# GIAO CHO: A
# NHIỆM VỤ: Implement hàm process_qa_sample (Task 2.1 – 2.5)
# INPUT:  question (str), context (str), answer (dict | None), tokenizer
# OUTPUT: input_ids, attention_mask, start_position, end_position (tensor)
###

from transformers import AutoTokenizer
import torch


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
    # Yêu cầu:
    #   - truncation="only_second"  → chỉ cắt context, không cắt question
    #   - stride=doc_stride         → dùng khi context dài (sliding window)
    #   - padding="max_length"      → pad đến max_length
    #   - return_overflowing_tokens=False  → chỉ lấy chunk đầu tiên
    #   - return_offsets_mapping=True      → cần để map char offset → token index
    #   - return_tensors="pt"
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

    input_ids      = inputs["input_ids"].squeeze(0)        # [max_length]
    attention_mask = inputs["attention_mask"].squeeze(0)   # [max_length]
    offset_mapping = inputs["offset_mapping"].squeeze(0)   # [max_length, 2]

    start_position = torch.tensor(0, dtype=torch.long)
    end_position   = torch.tensor(0, dtype=torch.long)

    # ------------------------------------------------------------------
    # TODO 2.2 — Xác định vùng context trong chuỗi token
    # ------------------------------------------------------------------
    # Dùng sequence_ids() để biết token nào thuộc question (id=0)
    # và token nào thuộc context (id=1).
    # Sau đó tìm:
    #   - token_start_index: index token đầu tiên của context
    #   - token_end_index  : index token cuối cùng của context
    #
    # Gợi ý:
    #   sequence_ids = inputs.sequence_ids(0)  # list[int | None]
    #   # sequence_ids[i] == 1  →  token i thuộc context
    #   # sequence_ids[i] == 0  →  token i thuộc question
    #   # sequence_ids[i] is None  →  special token ([CLS], [SEP])
    #
    # TODO: Viết vòng lặp tìm token_start_index và token_end_index
    # ...

    # ------------------------------------------------------------------
    # TODO 2.3 — Map character offset → token index (chỉ khi có answer)
    # ------------------------------------------------------------------
    # Chỉ thực hiện khi: answer is not None AND len(answer["answer_start"]) > 0
    #
    # Bước 1: Tính start_char và end_char từ answer dict
    #   start_char = answer["answer_start"][0]
    #   end_char   = start_char + len(answer["text"][0])
    #
    # Bước 2: Duyệt offset_mapping từ token_start_index → token_end_index
    #   - Tìm token i đầu tiên có offset_mapping[i][0] >= start_char  → start token
    #   - Tìm token j cuối cùng có offset_mapping[j][1] <= end_char   → end token
    #
    # Gợi ý xử lý offset:
    #   offset_mapping[i] = (char_start, char_end) của token thứ i
    #   Token special ([CLS], [SEP]) thường có offset (0, 0) → bỏ qua

    # ------------------------------------------------------------------
    # TODO 2.4 — Xử lý edge cases
    # ------------------------------------------------------------------
    # Trường hợp cần set start = end = 0 (unanswerable / bị truncate):
    #   1. answer is None
    #   2. answer["answer_start"] rỗng  (ViQuAD unanswerable)
    #   3. answer span nằm ngoài phần context còn lại sau khi truncate
    #      → Kiểm tra: start_char < offset_mapping[token_start_index][0]
    #                   hoặc end_char > offset_mapping[token_end_index][1]

    return input_ids, attention_mask, start_position, end_position


# 
# TODO 2.5 — Unit test
# 
if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base", use_fast=True)

    # --- Test 1: SQuAD — có answer rõ ràng ---
    ids, mask, s, e = process_qa_sample(
        question="When did the first World Cup happen?",
        context="The first World Cup was held in 1930 in Uruguay.",
        answer={"text": ["1930"], "answer_start": [28]},
        tokenizer=tokenizer,
    )
    print(f"[Test 1 - SQuAD]  start={s.item()}, end={e.item()}")
    # Kỳ vọng: s > 0, e >= s
    # Verify bằng cách decode lại token:
    print("  Answer tokens:", tokenizer.decode(ids[s : e + 1]))
    # Kỳ vọng output: "1930"

    # --- Test 2: ViQuAD — unanswerable (answer=None) ---
    ids, mask, s, e = process_qa_sample(
        question="Ai là tổng thống Việt Nam?",
        context="Việt Nam có thủ đô là Hà Nội.",
        answer=None,
        tokenizer=tokenizer,
    )
    print(f"[Test 2 - VI unanswerable]  start={s.item()}, end={e.item()}")
    # Kỳ vọng: s == 0, e == 0

    # --- Test 3: SQuAD — answer bị truncate (nằm ngoài max_length nhỏ) ---
    ids, mask, s, e = process_qa_sample(
        question="What is at the end?",
        context="Word " * 200 + "ANSWER is here.",
        answer={"text": ["ANSWER"], "answer_start": 1000},
        tokenizer=tokenizer,
        max_length=64,
    )
    print(f"[Test 3 - Truncated]  start={s.item()}, end={e.item()}")
    # Kỳ vọng: s == 0, e == 0  # preprocess_qa_sample.py
# ============================================================
# GIAO CHO: A
# NHIỆM VỤ: Implement hàm process_qa_sample (Task 2.1 – 2.5)
# INPUT:  question (str), context (str), answer (dict | None), tokenizer
# OUTPUT: input_ids, attention_mask, start_position, end_position (tensor)
# ============================================================

from transformers import AutoTokenizer
import torch


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
    # Yêu cầu:
    #   - truncation="only_second"  → chỉ cắt context, không cắt question
    #   - stride=doc_stride         → dùng khi context dài (sliding window)
    #   - padding="max_length"      → pad đến max_length
    #   - return_overflowing_tokens=False  → chỉ lấy chunk đầu tiên
    #   - return_offsets_mapping=True      → cần để map char offset → token index
    #   - return_tensors="pt"
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

    input_ids      = inputs["input_ids"].squeeze(0)        # [max_length]
    attention_mask = inputs["attention_mask"].squeeze(0)   # [max_length]
    offset_mapping = inputs["offset_mapping"].squeeze(0)   # [max_length, 2]

    start_position = torch.tensor(0, dtype=torch.long)
    end_position   = torch.tensor(0, dtype=torch.long)

    # ------------------------------------------------------------------
    # TODO 2.2 — Xác định vùng context trong chuỗi token
    # ------------------------------------------------------------------
    # Dùng sequence_ids() để biết token nào thuộc question (id=0)
    # và token nào thuộc context (id=1).
    # Sau đó tìm:
    #   - token_start_index: index token đầu tiên của context
    #   - token_end_index  : index token cuối cùng của context
    #
    # Gợi ý:
    #   sequence_ids = inputs.sequence_ids(0)  # list[int | None]
    #   # sequence_ids[i] == 1  →  token i thuộc context
    #   # sequence_ids[i] == 0  →  token i thuộc question
    #   # sequence_ids[i] is None  →  special token ([CLS], [SEP])
    #
    # TODO: Viết vòng lặp tìm token_start_index và token_end_index
    # ...

    # ------------------------------------------------------------------
    # TODO 2.3 — Map character offset → token index (chỉ khi có answer)
    # ------------------------------------------------------------------
    # Chỉ thực hiện khi: answer is not None AND len(answer["answer_start"]) > 0
    #
    # Bước 1: Tính start_char và end_char từ answer dict
    #   start_char = answer["answer_start"][0]
    #   end_char   = start_char + len(answer["text"][0])
    #
    # Bước 2: Duyệt offset_mapping từ token_start_index → token_end_index
    #   - Tìm token i đầu tiên có offset_mapping[i][0] >= start_char  → start token
    #   - Tìm token j cuối cùng có offset_mapping[j][1] <= end_char   → end token
    #
    # Gợi ý xử lý offset:
    #   offset_mapping[i] = (char_start, char_end) của token thứ i
    #   Token special ([CLS], [SEP]) thường có offset (0, 0) → bỏ qua

    # ------------------------------------------------------------------
    # TODO 2.4 — Xử lý edge cases
    # ------------------------------------------------------------------
    # Trường hợp cần set start = end = 0 (unanswerable / bị truncate):
    #   1. answer is None
    #   2. answer["answer_start"] rỗng  (ViQuAD unanswerable)
    #   3. answer span nằm ngoài phần context còn lại sau khi truncate
    #      → Kiểm tra: start_char < offset_mapping[token_start_index][0]
    #                   hoặc end_char > offset_mapping[token_end_index][1]

    return input_ids, attention_mask, start_position, end_position


# ============================================================
# TODO 2.5 — Unit test (chạy thẳng file này để kiểm tra)
# ============================================================
if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base", use_fast=True)

    # --- Test 1: SQuAD — có answer rõ ràng ---
    ids, mask, s, e = process_qa_sample(
        question="When did the first World Cup happen?",
        context="The first World Cup was held in 1930 in Uruguay.",
        answer={"text": ["1930"], "answer_start": [28]},
        tokenizer=tokenizer,
    )
    print(f"[Test 1 - SQuAD]  start={s.item()}, end={e.item()}")
    # Kỳ vọng: s > 0, e >= s
    # Verify bằng cách decode lại token:
    print("  Answer tokens:", tokenizer.decode(ids[s : e + 1]))
    # Kỳ vọng output: "1930"

    # --- Test 2: ViQuAD — unanswerable (answer=None) ---
    ids, mask, s, e = process_qa_sample(
        question="Ai là tổng thống Việt Nam?",
        context="Việt Nam có thủ đô là Hà Nội.",
        answer=None,
        tokenizer=tokenizer,
    )
    print(f"[Test 2 - VI unanswerable]  start={s.item()}, end={e.item()}")
    # Kỳ vọng: s == 0, e == 0

    # --- Test 3: SQuAD — answer bị truncate (nằm ngoài max_length nhỏ) ---
    ids, mask, s, e = process_qa_sample(
        question="What is at the end?",
        context="Word " * 200 + "ANSWER is here.",
        answer={"text": ["ANSWER"], "answer_start": 1000},
        tokenizer=tokenizer,
        max_length=64,
    )
    print(f"[Test 3 - Truncated]  start={s.item()}, end={e.item()}")
    # Kỳ vọng: s == 0, e == 0  (answer bị cắt mất)