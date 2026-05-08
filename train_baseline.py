# train_baseline.py
"""
Baseline: XLM-RoBERTa thuần túy cho QA (không có GAT, không có OT).

Dùng AutoModelForQuestionAnswering — đọc thẳng 512 tokens, dự đoán span.
Mục đích: so sánh với mô hình đề xuất (OT + Graph) trong Ablation Study.

Cách chạy:
    python train_baseline.py                       # default SQuAD 2.0
    python train_baseline.py --epochs 5 --batch_size 16
    python train_baseline.py --eval_only --checkpoint checkpoints_baseline/best_model

Evaluation trên ViQuAD2 (cross-lingual zero-shot):
    Sau khi train xong trên SQuAD (EN), dùng inference pipeline sẵn có
    để chạy trên ViQuAD2 dev set → đo F1/EM.
"""

import os
import sys
import math
import json
import argparse
import logging
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load .env
try:
    from dotenv import load_dotenv
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    load_dotenv(env_path)
except ImportError:
    pass

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Config mặc định
# ──────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "model_name"    : "xlm-roberta-base",
    "batch_size"    : 8,
    "grad_accum_steps": 4,
    "lr"            : 3e-5,
    "weight_decay"  : 0.01,
    "warmup_ratio"  : 0.06,
    "max_epochs"    : 3,
    "max_grad_norm" : 1.0,
    "max_length"    : 384,
    "doc_stride"    : 128,

    "root_dir"   : os.path.dirname(os.path.abspath(__file__)),
    "output_dir" : os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints_baseline"),
    "hf_repo_id" : "",
    "save_every" : 1,
    "log_every"  : 50,
}


# ──────────────────────────────────────────────────────────────
# Dataset: SQuAD-format → HuggingFace QA format
# ──────────────────────────────────────────────────────────────

class SQuADQADataset(Dataset):
    """
    Dataset cho bài toán QA truyền thống (token-level span extraction).
    Đọc trực tiếp từ file SQuAD-format JSON.
    """

    def __init__(self, data_file: str, tokenizer, max_length: int = 384, doc_stride: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.doc_stride = doc_stride
        self.samples = self._load_and_flatten(data_file)
        log.info(f"SQuADQADataset: {len(self.samples):,} samples từ {data_file}")

    def _load_and_flatten(self, file_path: str) -> list[dict]:
        """Flatten SQuAD-format JSON thành list[dict]."""
        with open(file_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        articles = raw["data"] if isinstance(raw, dict) and "data" in raw else raw
        samples = []
        for article in articles:
            for paragraph in article.get("paragraphs", []):
                context = paragraph["context"]
                for qa in paragraph.get("qas", []):
                    answers = qa.get("answers", [])
                    is_impossible = qa.get("is_impossible", len(answers) == 0)

                    if answers:
                        answer_text = answers[0]["text"]
                        answer_start = int(answers[0]["answer_start"])
                    else:
                        answer_text = ""
                        answer_start = -1

                    samples.append({
                        "id": qa.get("id", ""),
                        "question": qa["question"],
                        "context": context,
                        "answer_text": answer_text,
                        "answer_start": answer_start,
                        "is_impossible": is_impossible,
                    })
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        # Tokenize
        inputs = self.tokenizer(
            sample["question"],
            sample["context"],
            max_length=self.max_length,
            truncation="only_second",
            stride=self.doc_stride,
            padding="max_length",
            return_overflowing_tokens=False,
            return_offsets_mapping=True,
            return_tensors="pt",
        )

        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        offset_mapping = inputs["offset_mapping"].squeeze(0)

        # Mặc định: unanswerable → CLS token (index 0)
        start_position = 0
        end_position = 0

        if not sample["is_impossible"] and sample["answer_start"] >= 0:
            start_char = sample["answer_start"]
            end_char = start_char + len(sample["answer_text"])

            # Tìm vùng context (sequence_id == 1)
            sequence_ids = inputs.sequence_ids(0)
            token_start = 0
            while token_start < len(sequence_ids) and sequence_ids[token_start] != 1:
                token_start += 1
            token_end = len(sequence_ids) - 1
            while token_end >= 0 and sequence_ids[token_end] != 1:
                token_end -= 1

            # Kiểm tra answer có bị truncate không
            if token_start < len(offset_mapping) and token_end >= 0:
                ctx_start_offset = offset_mapping[token_start][0].item()
                ctx_end_offset = offset_mapping[token_end][1].item()

                if start_char >= ctx_start_offset and end_char <= ctx_end_offset:
                    # Tìm start token
                    s_idx = token_start
                    for i in range(token_start, token_end + 1):
                        if offset_mapping[i][0].item() <= start_char:
                            s_idx = i
                        else:
                            break

                    # Tìm end token
                    e_idx = token_end
                    for i in range(token_end, token_start - 1, -1):
                        if offset_mapping[i][1].item() >= end_char:
                            e_idx = i
                        else:
                            break

                    if s_idx <= e_idx:
                        start_position = s_idx
                        end_position = e_idx

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "start_positions": torch.tensor(start_position, dtype=torch.long),
            "end_positions": torch.tensor(end_position, dtype=torch.long),
        }


# ──────────────────────────────────────────────────────────────
# Training Loop
# ──────────────────────────────────────────────────────────────

def run_training(config: dict, device: torch.device):
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer

    log.info("=" * 60)
    log.info("BASELINE: XLM-RoBERTa for Question Answering (No OT, No GAT)")
    log.info("=" * 60)

    os.makedirs(config["output_dir"], exist_ok=True)

    # 1. Load model + tokenizer
    log.info(f"Loading model: {config['model_name']}...")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(config["model_name"]).to(device)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    log.info(f"Model: {total_params:.1f}M params | Device: {device}")

    # 2. Load dataset (SQuAD 2.0 — EN only)
    squad_file = os.path.join(config["root_dir"], "dataset", "Squad2.0", "train-v2.0.json")
    if not os.path.exists(squad_file):
        log.error(f"Không tìm thấy SQuAD file: {squad_file}")
        return

    dataset = SQuADQADataset(
        data_file=squad_file,
        tokenizer=tokenizer,
        max_length=config["max_length"],
        doc_stride=config["doc_stride"],
    )

    train_loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    log.info(f"DataLoader: {len(train_loader)} batches/epoch")

    # 3. Optimizer + Scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )

    steps_per_epoch = math.ceil(len(train_loader) / config["grad_accum_steps"])
    total_steps = steps_per_epoch * config["max_epochs"]
    warmup_steps = int(total_steps * config["warmup_ratio"])

    try:
        from transformers import get_linear_schedule_with_warmup
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        log.info(f"Scheduler: linear warmup {warmup_steps}/{total_steps} steps")
    except ImportError:
        scheduler = None
        log.warning("transformers scheduler không tìm thấy")

    # 4. Optional: TensorBoard
    try:
        from torch.utils.tensorboard import SummaryWriter
        tb_log_dir = os.path.join(config["output_dir"], "tensorboard_logs")
        writer = SummaryWriter(log_dir=tb_log_dir)
        log.info(f"TensorBoard: {tb_log_dir}")
    except ImportError:
        writer = None

    # 5. Optional: Resume
    start_epoch = 1
    global_step = 0
    if config.get("resume_from") and os.path.exists(config["resume_from"]):
        log.info(f"Resuming from {config['resume_from']}...")
        ckpt = torch.load(config["resume_from"], map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        if scheduler and ckpt.get("scheduler_state"):
            scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt.get("epoch", 0) + 1
        global_step = ckpt.get("global_step", 0)
        log.info(f"Resumed from epoch {ckpt.get('epoch')}, step {global_step}")

    # 6. Training loop
    best_loss = float("inf")
    optimizer.zero_grad()

    for epoch in range(start_epoch, config["max_epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        accum_count = 0

        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                start_positions=batch["start_positions"],
                end_positions=batch["end_positions"],
            )

            loss = outputs.loss / config["grad_accum_steps"]
            loss.backward()

            epoch_loss += outputs.loss.item()
            accum_count += 1

            if (step + 1) % config["grad_accum_steps"] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
                optimizer.step()
                if scheduler:
                    scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % config["log_every"] == 0:
                    log.info(
                        f"Epoch {epoch} | Step {global_step} | "
                        f"loss={outputs.loss.item():.4f} | "
                        f"lr={optimizer.param_groups[0]['lr']:.2e}"
                    )
                    if writer:
                        writer.add_scalar("Loss/Train", outputs.loss.item(), global_step)
                        writer.add_scalar("Learning_Rate", optimizer.param_groups[0]['lr'], global_step)

        avg_loss = epoch_loss / max(accum_count, 1)
        log.info(f"━━ Epoch {epoch}/{config['max_epochs']} done | avg_loss={avg_loss:.4f}")

        # Save checkpoint
        if epoch % config["save_every"] == 0:
            ckpt_path = os.path.join(config["output_dir"], f"epoch_{epoch:03d}.pt")
            save_dict = {
                "epoch": epoch,
                "global_step": global_step,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict() if scheduler else None,
                "config": config,
                "avg_loss": avg_loss,
            }
            torch.save(save_dict, ckpt_path)
            log.info(f"   Checkpoint saved: {ckpt_path}")

            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_path = os.path.join(config["output_dir"], "best_model")
                model.save_pretrained(best_path)
                tokenizer.save_pretrained(best_path)
                log.info(f"   🏆 New best model saved: {best_path} (loss={avg_loss:.4f})")

            # Upload to HuggingFace nếu có config
            if config.get("hf_repo_id"):
                try:
                    from huggingface_hub import HfApi
                    api = HfApi()
                    log.info(f"   Uploading to HuggingFace: {config['hf_repo_id']}...")
                    api.upload_file(
                        path_or_fileobj=ckpt_path,
                        path_in_repo=f"baseline_checkpoints/epoch_{epoch:03d}.pt",
                        repo_id=config["hf_repo_id"],
                        repo_type="model",
                    )
                    log.info(f"   ✅ Uploaded successfully!")
                except Exception as e:
                    log.error(f"   Upload error (file local vẫn còn): {e}")

    log.info("Baseline training hoàn thành!")
    if writer:
        writer.close()


# ──────────────────────────────────────────────────────────────
# Inference: Chạy model trên ViQuAD2 dev set → predictions.json
# ──────────────────────────────────────────────────────────────

def run_inference(config: dict, device: torch.device):
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer

    log.info("=" * 60)
    log.info("BASELINE INFERENCE: XLM-R on ViQuAD2 dev set")
    log.info("=" * 60)

    ckpt_path = config.get("checkpoint", os.path.join(config["output_dir"], "best_model"))
    if not os.path.exists(ckpt_path):
        log.error(f"Checkpoint không tìm thấy: {ckpt_path}")
        return

    log.info(f"Loading model từ {ckpt_path}...")
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(ckpt_path).to(device)
    model.eval()

    # Load ViQuAD2 dev set
    viquad_dev = config.get(
        "eval_file",
        os.path.join(config["root_dir"], "dataset", "ViQA", "ViQuAD_v2", "UIT-ViQuAD_2.0", "dev.json"),
    )
    if not os.path.exists(viquad_dev):
        # Thử thêm một vài path phổ biến
        alt_paths = [
            os.path.join(config["root_dir"], "dataset", "ViQA", "ViQuad v2", "UIT-ViQuAD 2.0", "dev (public_test).json"),
            os.path.join(config["root_dir"], "dataset", "ViQA", "dev.json"),
        ]
        for alt in alt_paths:
            if os.path.exists(alt):
                viquad_dev = alt
                break
        else:
            log.error(f"ViQuAD2 dev file không tìm thấy. Đã thử: {viquad_dev} + {alt_paths}")
            return

    log.info(f"Dev file: {viquad_dev}")

    with open(viquad_dev, "r", encoding="utf-8") as f:
        raw = json.load(f)

    articles = raw["data"] if isinstance(raw, dict) and "data" in raw else raw

    predictions = {}
    total = 0

    for article in articles:
        for paragraph in article.get("paragraphs", []):
            context = paragraph["context"]
            for qa in paragraph.get("qas", []):
                qid = qa["id"]
                question = qa["question"]

                inputs = tokenizer(
                    question,
                    context,
                    max_length=config["max_length"],
                    truncation="only_second",
                    stride=config["doc_stride"],
                    padding="max_length",
                    return_tensors="pt",
                )

                with torch.no_grad():
                    outputs = model(
                        input_ids=inputs["input_ids"].to(device),
                        attention_mask=inputs["attention_mask"].to(device),
                    )

                start_idx = outputs.start_logits.argmax(dim=-1).item()
                end_idx = outputs.end_logits.argmax(dim=-1).item()

                if start_idx <= end_idx and start_idx > 0:
                    answer_ids = inputs["input_ids"][0][start_idx:end_idx + 1]
                    answer = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
                else:
                    answer = ""

                predictions[qid] = answer
                total += 1

    # Save predictions
    pred_path = os.path.join(config["output_dir"], "baseline_predictions.json")
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    log.info(f"Đã lưu {total} predictions → {pred_path}")
    log.info(f"Dùng evaluate_json_pipeline.py --mode squad2 --squad_file {viquad_dev} --pred_file {pred_path}")


# ──────────────────────────────────────────────────────────────
# Entrypoint
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Baseline: XLM-R QA (No OT, No GAT)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--mode", choices=["train", "eval"], default="train",
                        help="train: train trên SQuAD 2.0\neval: inference trên ViQuAD2 dev")
    parser.add_argument("--root_dir", type=str, default=DEFAULT_CONFIG["root_dir"])
    parser.add_argument("--output_dir", type=str, default=DEFAULT_CONFIG["output_dir"])
    parser.add_argument("--hf_repo_id", type=str, default=DEFAULT_CONFIG["hf_repo_id"],
                        help="HuggingFace repo để auto backup")
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["max_epochs"])
    parser.add_argument("--batch_size", type=int, default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG["lr"])
    parser.add_argument("--max_length", type=int, default=DEFAULT_CONFIG["max_length"])
    parser.add_argument("--checkpoint", type=str, default="",
                        help="Path to checkpoint for eval mode")
    parser.add_argument("--eval_file", type=str, default="",
                        help="Path to ViQuAD2 dev file for eval mode")
    parser.add_argument("--resume_from", type=str, default="",
                        help="Path to checkpoint to resume training")
    args = parser.parse_args()

    config = dict(DEFAULT_CONFIG)
    config.update({
        "root_dir": args.root_dir,
        "output_dir": args.output_dir,
        "hf_repo_id": args.hf_repo_id,
        "max_epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "max_length": args.max_length,
        "resume_from": args.resume_from,
    })
    if args.checkpoint:
        config["checkpoint"] = args.checkpoint
    if args.eval_file:
        config["eval_file"] = args.eval_file

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        log.info(f"✅ CUDA | Device: {torch.cuda.get_device_name(0)}")
    else:
        log.info(f"Device: {device}")

    if args.mode == "train":
        run_training(config, device)
    else:
        run_inference(config, device)


if __name__ == "__main__":
    main()
