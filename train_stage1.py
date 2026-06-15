import os
import sys
import argparse
import logging
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from phase1_dataloader.process_qa_sample import load_squad_data, process_qa_sample
from phase2_model.model_core import CrossLingualOTModel
from phase3_loss.losses import OTAlignmentLoss

logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO, datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

class SquadDatasetStage1(Dataset):
    def __init__(self, squad_file, tokenizer, max_length=384, doc_stride=128):
        self.data = load_squad_data(squad_file)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.doc_stride = doc_stride

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        is_answerable = len(sample["answer"]["answer_start"]) > 0
        input_ids, attention_mask, start_pos, end_pos, q_end = process_qa_sample(
            question=sample["question"],
            context=sample["context"],
            answer=sample["answer"] if is_answerable else None,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            doc_stride=self.doc_stride
        )
        return {
            "en_input_ids": input_ids,
            "en_attention_mask": attention_mask,
            "en_start_position": start_pos,
            "en_end_position": end_pos,
            "en_question_end": q_end,
            "en_is_answerable": torch.tensor(1 if is_answerable else 0, dtype=torch.long)
        }

def run_stage1(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Starting Stage 1 on {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], use_fast=True)
    dataset = SquadDatasetStage1(config["squad_file"], tokenizer, max_length=384, doc_stride=128)
    
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, num_workers=2, pin_memory=True)
    
    # Model (compute_cost_matrix=False for speed)
    model = CrossLingualOTModel(model_name=config["model_name"], compute_cost_matrix=False).to(device)
    
    # Loss — OT/span/cons đều tắt (λ=0), chỉ L_qa + L_has_ans hoạt động.
    # L_qa train trên toàn bộ batch (unanswerable -> start=0, end=0).
    # L_has_ans train trên toàn bộ batch (BCE).
    # → Model sẽ học cách predict [CLS] (index 0) cho unanswerable.
    criterion = OTAlignmentLoss(
        hidden_size=model.hidden_size,
        lambda_ot=0.0,
        lambda_span=0.0,
        lambda_cons=0.0,
    ).to(device)
    
    optimizer = AdamW([
        {"params": model.backbone.parameters(), "lr": config["lr"]},
        {"params": [model.layer_weights], "lr": config["head_lr"]},
        {"params": criterion.parameters(), "lr": config["head_lr"]},
    ], weight_decay=config["weight_decay"])
    
    # Scheduler
    total_steps = len(loader) * config["epochs"]
    warmup_steps = int(total_steps * 0.08)
    try:
        from transformers import get_linear_schedule_with_warmup
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    except:
        scheduler = None

    best_em = 0.0
    os.makedirs(os.path.join(config["root_dir"], "checkpoints"), exist_ok=True)
    
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        criterion.train()
        epoch_loss = 0.0
        
        for step, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # EN-only forward pass (branch="en" bỏ qua VI encoder hoàn toàn)
            outputs = model(batch, branch="en")
            
            B, L, H = outputs["hidden"].shape

            # Dummy VI branch — Stage 1 chỉ train EN, không có VI data thật.
            # Shape (B, 2, H): đủ để vi_question_end=1 không out-of-bounds.
            # dummy_vi_mask = toàn True (PAD) để SWD/Sinkhorn bỏ qua VI tokens.
            dummy_vi      = torch.zeros(B, 2, H, device=device)
            dummy_vi_mask = torch.ones(B, 2, dtype=torch.bool, device=device)
            batch["vi_question_end"] = torch.ones(B, dtype=torch.long, device=device)

            model_outputs = {
                "en_hidden"   : outputs["hidden"],
                "vi_hidden"   : dummy_vi,
                "en_pad_mask" : outputs["en_pad_mask"],
                "vi_pad_mask" : dummy_vi_mask,
                "cost_matrix" : torch.zeros(B, 2, 2, device=device),
            }

            # L_qa train trên toàn bộ batch (kể cả unanswerable)       ✓
            # losses["has_ans"] = BCE   trên toàn batch            ✓
            # losses["ot"]      = 0.0   vì lambda_ot=0.0           ✓
            # losses["total"]   = L_qa + L_has_ans                 ✓
            losses = criterion(model_outputs, batch)
            
            loss = losses["total"]
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(criterion.parameters(), 1.0)
            optimizer.step()
            if scheduler: scheduler.step()
            optimizer.zero_grad()
            
            epoch_loss += loss.item()
            
            if step % 50 == 0:
                ans_count = batch["en_is_answerable"].sum().item()
                log.info(
                    f"Epoch {epoch} Step {step}/{len(loader)} | "
                    f"Loss: {loss.item():.4f} "
                    f"(QA: {losses['qa'].item():.4f}, "
                    f"HasAns: {losses['has_ans'].item():.4f}, "
                    f"answerable: {ans_count}/{B})"
                )
                
        log.info(f"━━ Epoch {epoch} Avg Loss: {epoch_loss/len(loader):.4f} ━━")
        
        # Eval
        import importlib.util
        eval_file = os.path.join(config["root_dir"], "phase4-evaluation", "quick_eval.py")
        spec = importlib.util.spec_from_file_location("quick_eval", eval_file)
        quick_eval_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(quick_eval_mod)
        
        dev_file = os.path.join(config["root_dir"], "dataset", "Squad2.0", "dev-v2.0.json")
        try:
            em = quick_eval_mod.quick_em(model, criterion, tokenizer, dev_file, n_samples=500, device=device)
            log.info(f"Epoch {epoch} Quick EM (500 samples): {em:.2f}%")
        except Exception as e:
            log.error(f"Eval failed: {e}")
            em = 0.0
            
        # Save best model
        if em >= best_em:
            best_em = em
            save_path = os.path.join(config["root_dir"], "checkpoints", "stage1_squad_best.pt")
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "criterion_state": criterion.state_dict(),
                "em": em,
            }, save_path)
            log.info(f"🏆 Saved best Stage 1 checkpoint to {save_path} (EM: {em:.2f}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", "--epoch", dest="epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--hf_repo_id", type=str, default="")
    args = parser.parse_args()

    config = {
        "model_name": "xlm-roberta-base",
        "squad_file": os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset/Squad2.0/train-v2.0.json"),
        "root_dir": os.path.dirname(os.path.abspath(__file__)),
        "batch_size": args.batch_size,
        "lr": args.lr,
        "head_lr": 8e-5,
        "weight_decay": 0.01,
        "epochs": args.epochs,
        "mode": args.mode,
        "hf_repo_id": args.hf_repo_id
    }
    run_stage1(config)