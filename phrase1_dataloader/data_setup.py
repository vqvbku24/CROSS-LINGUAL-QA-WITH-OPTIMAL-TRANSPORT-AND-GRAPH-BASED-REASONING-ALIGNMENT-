# data_setup.py
"""
Setup dataset và tokenizer cho Phase 1.
Load từ LOCAL files trên Google Drive — không cần kết nối HuggingFace cho data.

Cấu trúc thư mục giả định:
    MyDrive/
    └── CROSS-LINGUAL-.../
        └── dataset/
            ├── Squad2.0/
            │   └── train-v2.0.json
            └── ViQA/
                └── ViQuad v2/
                    └── UIT-ViQuAD 2.0/
                        └── train.json

FIX so với phiên bản trước:
    1. flatten_squad dùng sai với load_dataset().map() → crash
       Giờ dùng _flatten_squad_json() đọc JSON thuần, không phụ thuộc HuggingFace datasets
    2. ViQuAD local cùng format SQuAD → flatten chung 1 hàm, format đồng nhất
    3. ListDataset wrapper để tương thích CrossLingualQADataset (ds[i], ds[i].get("title"))
    4. FileNotFoundError rõ ràng thay vì crash ngầm
    5. answer_start luôn là list[int] — đúng format process_qa_sample mong đợi
"""

import os
import json
import torch
from torch.utils.data import Dataset


# ──────────────────────────────────────────────────────────────
# Mock function
# ──────────────────────────────────────────────────────────────
def mock_process_qa_sample(
    question: str,
    context: str,
    answer=None,
    tokenizer=None,
    max_length: int = 512,
    doc_stride: int = 128,
):
    input_ids      = torch.randint(0, 25000, (max_length,), dtype=torch.long)
    attention_mask = torch.ones(max_length, dtype=torch.long)
    start_position = torch.tensor(0, dtype=torch.long)
    end_position   = torch.tensor(0, dtype=torch.long)
    question_end   = torch.tensor(5, dtype=torch.long)
    return input_ids, attention_mask, start_position, end_position, question_end


# ──────────────────────────────────────────────────────────────
# Core helper: flatten SQuAD-format JSON → list[dict]
# ──────────────────────────────────────────────────────────────
def _flatten_squad_json(file_path: str) -> list:
    """
    Đọc file JSON dạng SQuAD (data → paragraphs → qas) và flatten thành
    list phẳng, mỗi phần tử là 1 câu hỏi:

        {
            "id"      : str,
            "title"   : str,
            "question": str,
            "context" : str,
            "answers" : {
                "text"        : list[str],
                "answer_start": list[int]
            }
        }

    Hoạt động với cả SQuAD 2.0 và ViQuAD 2.0 (cùng format).
    """
    with open(file_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    articles = raw["data"] if isinstance(raw, dict) and "data" in raw else raw

    samples = []
    for article in articles:
        title = article.get("title", "")
        for paragraph in article.get("paragraphs", []):
            context = paragraph["context"]
            for qa in paragraph.get("qas", []):
                raw_answers = qa.get("answers", [])
                if raw_answers:
                    answer_dict = {
                        "text":         [a["text"] for a in raw_answers],
                        "answer_start": [int(a["answer_start"]) for a in raw_answers],
                    }
                else:
                    answer_dict = {"text": [], "answer_start": []}

                samples.append({
                    "id":       qa.get("id", ""),
                    "title":    title,
                    "question": qa["question"],
                    "context":  context,
                    "answers":  answer_dict,
                })

    return samples


# ──────────────────────────────────────────────────────────────
# Dataset wrapper
# ──────────────────────────────────────────────────────────────
class ListDataset(Dataset):
    """
    Wrap list[dict] thành torch Dataset.
    CrossLingualQADataset gọi ds[i] và ds[i].get("title", "").
    """

    def __init__(self, data: list):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        return self.data[idx]


# ──────────────────────────────────────────────────────────────
# Lazy loader chính
# ──────────────────────────────────────────────────────────────
def get_setup_objects(
  root_dir = "/content/drive/MyDrive/CROSS-LINGUAL-QA-WITH-OPTIMAL-TRANSPORT-AND-GRAPH-BASED-REASONING-ALIGNMENT-",
  teacher_file="dataset/Squad2.0/train-v2.0.json",
  student_file="dataset/ViQA/ViQuAD_v2/UIT-ViQuAD_2.0/train.json",
  model_name="xlm-roberta-base"):
    """
    Load dataset local + tokenizer.

    Returns:
        teacher_dataset : ListDataset (SQuAD 2.0, EN)
        student_dataset : ListDataset (ViQuAD 2.0, VI)
        tokenizer       : XLM-R fast tokenizer
    """
    from transformers import AutoTokenizer

    teacher_path = os.path.join(root_dir, teacher_file)
    student_path = os.path.join(root_dir, student_file)

    # Kiểm tra file tồn tại
    for path, label in [(teacher_path, "Teacher SQuAD"), (student_path, "Student ViQuAD")]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"[{label}] Không tìm thấy:\n  {path}\n"
                f"Kiểm tra lại root_dir: {root_dir}"
            )

    print("Loading Teacher (SQuAD 2.0) từ local...")
    teacher_dataset = ListDataset(_flatten_squad_json(teacher_path))
    print(f"  → {len(teacher_dataset):,} samples")

    print("Loading Student (ViQuAD 2.0) từ local...")
    student_dataset = ListDataset(_flatten_squad_json(student_path))
    print(f"  → {len(student_dataset):,} samples")

    print("\n--- Teacher sample[0] ---")
    s = teacher_dataset[0]
    print(f"  title    : {s['title']}")
    print(f"  question : {s['question'][:80]}")
    print(f"  context  : {s['context'][:60]}...")
    print(f"  answers  : {s['answers']}")

    print("\n--- Student sample[0] ---")
    s = student_dataset[0]
    print(f"  title    : {s['title']}")
    print(f"  question : {s['question'][:80]}")
    print(f"  context  : {s['context'][:60]}...")
    print(f"  answers  : {s['answers']}")

    print(f"\nLoading tokenizer: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    print(f"  Vocab size : {tokenizer.vocab_size:,}")
    print(f"  [CLS] id   : {tokenizer.cls_token_id}")
    print(f"  [SEP] id   : {tokenizer.sep_token_id}")

    return teacher_dataset, student_dataset, tokenizer


# ──────────────────────────────────────────────────────────────
# Backward compatibility
# ──────────────────────────────────────────────────────────────
def _load_setup_objects():
    return get_setup_objects()


# ──────────────────────────────────────────────────────────────
# Self-test: python data_setup.py
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    teacher_dataset, student_dataset, tokenizer = get_setup_objects()

    print("\n--- Kiểm tra answer_start format ---")
    errors = []
    for i in range(min(50, len(teacher_dataset))):
        s   = teacher_dataset[i]
        ans = s["answers"]
        if not isinstance(ans["answer_start"], list):
            errors.append(f"index {i}: answer_start không phải list")
        if ans["answer_start"] and not isinstance(ans["answer_start"][0], int):
            errors.append(f"index {i}: answer_start[0] không phải int")
    if errors:
        for e in errors:
            print(f"  ✗ {e}")
    else:
        print(f"  ✓ 50 samples đầu — answer_start đều là list[int]")

    print("\n--- Kiểm tra title field (topic pairing) ---")
    titles = {teacher_dataset[i]["title"] for i in range(min(50, len(teacher_dataset)))}
    print(f"  ✓ {len(titles)} unique titles trong 50 samples: {sorted(titles)[:3]}...")

    unanswerable = sum(
        1 for i in range(len(teacher_dataset))
        if not teacher_dataset[i]["answers"]["answer_start"]
    )
    print(f"\n  SQuAD unanswerable: {unanswerable:,} / {len(teacher_dataset):,} "
          f"({unanswerable/len(teacher_dataset)*100:.1f}%)")

    print("\ndata_setup OK!")