# data_setup.py
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import random

# ==================== TASK 1.2 & 1.3: Load dataset + Tokenizer ====================
print("Đang load dataset...")
teacher_dataset = load_dataset("squad_v2", split="train")          # English
student_dataset = load_dataset("uit-nlp/viquad", split="train")    # Vietnamese

print(f"Teacher (EN) size: {len(teacher_dataset):,}")
print(f"Student (VI) size: {len(student_dataset):,}")

model_name = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)  # BẮT BUỘC use_fast=True

# ==================== TASK 1.4: Mock function (để B code song song) ====================
def mock_process_qa_sample(question, context, answer=None):
    """MOCK function - Vinh viết tạm để B có thể code __getitem__ ngay"""
    # Trả về tensor random có shape đúng
    input_ids = torch.randint(0, 25000, (512,))
    attention_mask = torch.ones(512, dtype=torch.long)
    start_pos = torch.tensor(0, dtype=torch.long)
    end_pos = torch.tensor(0, dtype=torch.long)
    return input_ids, attention_mask, start_pos, end_pos

# Sau này sẽ replace bằng hàm thật của A
# process_qa_sample = mock_process_qa_sample