"""
debug_overfit.py — chạy 1 forward pass và in diagnostics.
Usage: python debug_overfit.py
"""
import torch
import torch.nn as nn

# Setup
from phrase1_dataloader.data_setup import get_setup_objects
from phrase1_dataloader.cross_lingual_dataset import create_dataloader
from phrase2_model.model_core import CrossLingualOTModel
from phrase3_loss.losses import OTAlignmentLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

# Data
teacher_ds, student_ds, tokenizer = get_setup_objects(root_dir="./")
loader = create_dataloader(teacher_ds, student_ds, tokenizer,
                           batch_size=4, shuffle=True,
                           pairing_strategy="topic", num_workers=0)
batch = next(iter(loader))
batch = {k: v.to(device) for k, v in batch.items()}

print("=== BATCH LABELS ===")
print(f"  en_start_position : {batch['en_start_position'].tolist()}")
print(f"  en_end_position   : {batch['en_end_position'].tolist()}")
print(f"  en_question_end   : {batch['en_question_end'].tolist()}")
print()

# Model
model = CrossLingualOTModel(K=160, gat_hidden=512, gat_out=256).to(device)
criterion = OTAlignmentLoss(qa_hidden_size=256, K=160).to(device)

# Forward
model.train(); criterion.train()
raw = model(batch)

en_emb = raw["en_node_emb"]   # (B, K, 256)
vi_emb = raw["vi_node_emb"]
keep   = raw["keep_idx_en"]   # (B, K)

print("=== EMBEDDING DIAGNOSTICS ===")
for b in range(en_emb.size(0)):
    e = en_emb[b]  # (K, 256)
    std_across_nodes = e.std(dim=0).mean().item()   # std across K nodes per dim → mean
    std_across_dim   = e.std(dim=1).mean().item()   # std across dims per node
    print(f"  Sample {b}: std_across_nodes={std_across_nodes:.6f}  std_across_dim={std_across_dim:.6f}")
print()

print("=== KEEP_IDX_EN (first 10 of each sample) ===")
for b in range(keep.size(0)):
    print(f"  Sample {b}: {keep[b, :10].tolist()} ...")
print()

# Check if answer tokens are in keep
from phrase3_loss.losses import _remap_positions_to_graph_space
gs_start, gs_end = _remap_positions_to_graph_space(
    batch["en_start_position"], batch["en_end_position"], keep
)
print("=== LABEL REMAP ===")
for b in range(4):
    s_tok = batch["en_start_position"][b].item()
    e_tok = batch["en_end_position"][b].item()
    exact_match_s = (keep[b] == s_tok).any().item()
    exact_match_e = (keep[b] == e_tok).any().item()
    print(f"  Sample {b}: token({s_tok},{e_tok}) → graph({gs_start[b].item()},{gs_end[b].item()}) "
          f"| exact_s={exact_match_s} exact_e={exact_match_e}")
print()

# Check adjacency matrix sparsity
print("=== ADJACENCY MATRIX (en_sub) SPARSITY ===")
# Re-run subsampling to get adj
from phrase2_model.modules.subsampling import conditional_subsample
from phrase2_model.modules.backbone import SharedBackbone
backbone = SharedBackbone().to(device)
with torch.no_grad():
    en_hidden, en_attn = backbone(batch["en_input_ids"], batch["en_attention_mask"])
for b in range(4):
    q_end = batch["en_question_end"][b].item()
    s = batch["en_start_position"][b].item()
    e = batch["en_end_position"][b].item()
    q_idx = list(range(0, q_end + 1))
    a_idx = list(range(s, e + 1)) if s > 0 else []
    sub, kidx = conditional_subsample(en_attn[b], q_idx, a_idx, K=160)
    nonzero_frac = (sub > 0).float().mean().item()
    attn_min = sub.min().item()
    attn_max = sub.max().item()
    print(f"  Sample {b}: adj non-zero={nonzero_frac:.4f} | min={attn_min:.6f} max={attn_max:.6f}")
print()

# Loss + backward
from phrase3_loss.train import _patch_model_outputs
outputs = _patch_model_outputs(model, batch, raw)
losses = criterion(outputs, batch)
losses["total"].backward()

print("=== GRADIENT NORMS ===")
qa_grad = 0.0
for name, p in criterion.qa_head.named_parameters():
    if p.grad is not None:
        g = p.grad.norm().item()
        qa_grad += g
        print(f"  qa_head.{name}: grad_norm={g:.8f}")
print(f"  → Total qa_head grad: {qa_grad:.8f}")
print()

gat_grad = sum(p.grad.norm().item() for p in model.gat.parameters() if p.grad is not None)
bb_grad  = sum(p.grad.norm().item() for p in model.backbone.parameters() if p.grad is not None)
print(f"  GAT grad_norm total   : {gat_grad:.6f}")
print(f"  Backbone grad_norm total: {bb_grad:.6f}")
