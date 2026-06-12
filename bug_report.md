# ABLATION_SPEC.md
# Ablation: Freeze QA Head + λ_span=0 — Pure OT Backbone Alignment
# Mục đích: Kiểm tra xem L_span có phải thủ phạm phá hoại VI EM không.
# Chỉ cần 2 thay đổi trong train_stage2.py. Không đụng losses.py hay xquad_loader.py.

---

## Bối cảnh

Kết quả hiện tại:
  Stage 1 zero-shot VI EM = 56.67%  ← baseline thực sự
  Stage 2 epoch 1 VI EM  = 36.67%  ← tệ hơn baseline
  Stage 2 epoch 2 VI EM  = 12.78%  ← collapse hoàn toàn

Giả thuyết cần kiểm tra:
  L_span (KL pseudo-label) phá hoại QA Head bằng cách ép Head
  fit pseudo-labels noisy từ γ → Head mất EN prior tốt từ Stage 1.
  Nếu đúng: chỉ dùng L_ot + L_cons để update backbone → VI EM tăng.

---

## THAY ĐỔI 1: Thêm freeze_qa_head vào train_stage2.py

Thêm hàm sau ngay sau hàm freeze_en_backbone() (line ~127):

```python
def freeze_qa_head(criterion):
    """Freeze QA Head: disable gradients entirely."""
    for p in criterion.qa_head.parameters():
        p.requires_grad_(False)
    log.info("QA head frozen (requires_grad=False)")


def unfreeze_qa_head(criterion):
    """Unfreeze QA Head (for future use)."""
    for p in criterion.qa_head.parameters():
        p.requires_grad_(True)
    log.info("QA head unfrozen")
```

Gọi freeze_qa_head() ngay sau freeze_en_backbone() trong run_stage2():

```python
freeze_en_backbone(model)
if config.get("freeze_qa_head", False):
    freeze_qa_head(criterion)
```

---

## THAY ĐỔI 2: Sửa optimizer khi QA Head frozen

Khi QA Head frozen, criterion.parameters() không có grad → không cần
đưa vào optimizer (AdamW với params không có grad sẽ raise warning hoặc
waste memory). Sửa optimizer block trong run_stage2():

```python
# Optimizer — thay đổi có điều kiện
if config.get("freeze_qa_head", False):
    # QA head frozen: chỉ update layer_weights
    # (backbone frozen với lr=0.0, head frozen, chỉ layer_weights trainable)
    optimizer = AdamW([
        {"params": list(model.backbone.parameters()), "lr": 0.0},
        {"params": [model.layer_weights],             "lr": config["stage2_head_lr"]},
        # criterion.parameters() KHÔNG có trong optimizer — đã frozen
    ], weight_decay=config["weight_decay"])
    log.info("Optimizer: only layer_weights trainable (QA head frozen)")
else:
    # Giữ nguyên optimizer cũ
    optimizer = AdamW([
        {"params": list(model.backbone.parameters()),  "lr": 0.0},
        {"params": [model.layer_weights],              "lr": config["stage2_head_lr"]},
        {"params": list(criterion.parameters()),       "lr": config["stage2_head_lr"]},
    ], weight_decay=config["weight_decay"])
```

QUAN TRỌNG: layer_weights vẫn được update. Nó điều chỉnh weighted
combination của 12 XLM-R layers → backbone VI vẫn có thể adapt nhẹ
mà không phá hoại QA Head.

---

## THAY ĐỔI 3: Thêm CLI argument

Thêm vào parse_args():

```python
parser.add_argument("--freeze_qa_head", action="store_true", default=False,
                    help="Freeze QA head (ablation: pure OT backbone alignment)")
```

---

## THAY ĐỔI 4: Sửa en_em_safety

Nới threshold từ 5.0 lên 20.0 để tránh hard stop sớm:

```python
parser.add_argument("--en_em_safety", type=float, default=STAGE2_CONFIG["en_em_safety"])
```

Đã có CLI arg rồi — chỉ cần truyền từ command line.

---

## KHÔNG THAY ĐỔI

- losses.py — giữ nguyên toàn bộ
- xquad_loader.py — giữ nguyên
- Stage2Loss — lambda_span=0.0 truyền từ CLI, không cần sửa code
- stage2_step() — giữ nguyên, L_span vẫn được tính nhưng
  Stage2Loss sẽ nhân với lambda_span=0.0 nên không có effect

---

## VERIFICATION CHECK trước khi chạy

```python
# Verify QA head frozen
assert not any(p.requires_grad for p in criterion.qa_head.parameters()), \
    "BUG: QA head vẫn có requires_grad=True"

# Verify backbone frozen
assert not any(p.requires_grad for p in model.backbone.parameters()), \
    "BUG: EN backbone vẫn có requires_grad=True"

# Verify layer_weights trainable
assert model.layer_weights.requires_grad, \
    "BUG: layer_weights bị frozen"

print("Ablation setup verified")
```

---

## COMMAND LINE

```bash
python train_stage2.py \
  --stage1_ckpt checkpoints/stage1_squad_best.pt \
  --freeze_qa_head \
  --lambda_span 0.0 \
  --lambda_ot 1.0 \
  --lambda_cons 0.5 \
  --epsilon 0.3 \
  --sinkhorn_iters 200 \
  --en_em_safety 20.0 \
  --stage2_head_lr 1e-5 \
  --max_epochs 5 \
  --log_every 32
```

LÝ DO các hyperparameter:
  - epsilon=0.3: cân bằng giữa 0.1 (row_err cao) và 0.5 (entropy quá cao)
  - sinkhorn_iters=200: đủ để converge với non-uniform mu
  - stage2_head_lr=1e-5: chỉ update layer_weights, không cần lr cao
  - max_epochs=5: đủ để thấy trend, tránh overrun

---

## KẾT QUẢ KỲ VỌNG VÀ DIỄN GIẢI

Scenario A — VI EM tăng lên > 56.67%:
  L_span là thủ phạm. OT backbone alignment có giá trị thực sự.
  → Paper contribution: "Feature-level OT alignment improves zero-shot
    transfer; prediction-level pseudo-labeling is harmful with noisy γ"
  → Tiếp tục: thêm epochs, tune epsilon/lambda_ot

Scenario B — VI EM giữ nguyên ~56% hoặc giảm:
  XLM-R đã aligned quá tốt, OT không có thêm signal dù ở feature level.
  → Paper contribution đổi framing sang Hướng C (joint training Stage 1)
  → Kết luận: post-hoc OT fails trên strong multilingual backbone

Dù kết quả nào xảy ra, đây là ablation study có giá trị cho paper.

---

## NO-TOUCH ZONES

  - losses.py (toàn bộ)
  - xquad_loader.py
  - model_core.py
  - train.py (Stage 1)
  - Stage 1 checkpoint files

---
*End of ABLATION_SPEC.md*