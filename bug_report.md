# LREG_SPEC.md — Thêm English Consistency Regularisation (L_Reg) vào Stage 2

**Target agent:** Claude Opus 4.7  
**Mục tiêu:** Implement `L_Reg = MSE(h_en_lora, h_en_frz)` theo paper ACL để ngăn
LoRA shared weights làm hỏng EN branch khi train VI. Đây là nguyên nhân gốc của
EN EM collapse (83% → 63%) và VI EM collapse (42% → 4%) trong các run hiện tại.

**Thay đổi cần thiết:** 3 files — `losses.py`, `train_stage2.py`, `model_core.py` (tuỳ chọn).  
**Không được đụng vào:** `quick_eval.py`, `xquad_loader.py`, Stage 1 code trong `losses.py`
(trên dòng 680), `save_stage2_checkpoint()`, HuggingFace upload blocks, `sinkhorn_masked()`.

---

## Tổng quan kiến trúc thay đổi

### Hiện tại (2 forward pass):
```
1. EN branch — LoRA OFF, no_grad  → h_en_frz  (frozen anchor)
2. VI branch — LoRA ON,  with_grad → h_vi
Loss = λ_ot * L_ot + λ_span * L_span + λ_cons * L_cons
```

### Sau khi fix (3 forward pass, theo paper):
```
1. EN branch — LoRA OFF, no_grad  → h_en_frz  (frozen anchor, không đổi)
2. EN branch — LoRA ON,  with_grad → h_en_lora (NEW — để tính L_Reg)
3. VI branch — LoRA ON,  with_grad → h_vi      (không đổi)
Loss = λ_ot * L_ot + λ_reg * L_reg
     (bỏ L_span và L_cons khỏi total — xem chi tiết bên dưới)
```

`L_Reg = mean over valid tokens of ||h_en_lora - h_en_frz||²`

Gradient của `L_Reg` chảy qua `h_en_lora` → LoRA weights → giữ EN representation
không bị drift khi VI gradients update shared LoRA params.

---

## Thay đổi 1: `losses.py` — Thêm hàm `compute_reg_loss`

**Vị trí:** Thêm hàm mới **sau dòng 902** (sau `compute_cons_loss`), trước `gamma_entropy`.  
**Không sửa** bất kỳ hàm nào hiện có.

```python
def compute_reg_loss(
    h_en_lora: torch.Tensor,   # (B, T_en, D) — EN hidden states WITH LoRA (has grad)
    h_en_frz: torch.Tensor,    # (B, T_en, D) — EN hidden states WITHOUT LoRA (no grad)
    en_mask: torch.Tensor,     # (B, T_en) — True = real token
) -> torch.Tensor:
    """
    L_Reg = mean per-token L2 deviation between trainable and frozen EN branches.

    Prevents LoRA shared weights from distorting EN representations while
    training on VI. Gradient flows through h_en_lora only (h_en_frz is detached).

    Per-token squared L2 distance, averaged over valid (non-PAD) tokens and batch.

    Args:
        h_en_lora : (B, T_en, D) — EN forward pass WITH LoRA adapters active
        h_en_frz  : (B, T_en, D) — EN forward pass with LoRA disabled (frozen anchor)
        en_mask   : (B, T_en) bool — True for real tokens, False for PAD

    Returns:
        scalar L_Reg
    """
    # Align sequence lengths (dynamic truncation may differ between the two passes)
    T = min(h_en_lora.size(1), h_en_frz.size(1))
    h_en_lora = h_en_lora[:, :T, :]
    h_en_frz  = h_en_frz[:, :T, :].detach()   # target is fixed — must detach
    mask      = en_mask[:, :T]                  # (B, T)

    # Per-token squared L2: (B, T)
    sq_diff = ((h_en_lora - h_en_frz) ** 2).sum(dim=-1)  # (B, T)

    # Mask PAD tokens and average over valid positions
    sq_diff = sq_diff * mask.float()
    n_valid = mask.float().sum().clamp(min=1.0)
    return sq_diff.sum() / n_valid
```

---

## Thay đổi 2: `losses.py` — Sửa `Stage2Loss`

**Vị trí:** Class `Stage2Loss` từ dòng 925.

### 2a. Sửa `__init__`

**Tìm:**
```python
    def __init__(
        self,
        lambda_ot: float   = 1.0,
        lambda_span: float = 1.0,
        lambda_cons: float = 0.5,
    ):
        """
        Args:
            lambda_ot    : weight for L_ot (transport cost)
            lambda_span  : weight for L_span (pseudo-label KL)
            lambda_cons  : weight for L_cons (feature consistency MSE)
        """
        super().__init__()
        self.lambda_ot    = lambda_ot
        self.lambda_span  = lambda_span
        self.lambda_cons  = lambda_cons
```

**Thay bằng:**
```python
    def __init__(
        self,
        lambda_ot: float   = 1.0,
        lambda_span: float = 0.0,
        lambda_cons: float = 0.0,
        lambda_reg: float  = 10.0,
    ):
        """
        Args:
            lambda_ot    : weight for L_ot (transport cost). Default 1.0.
            lambda_span  : weight for L_span. Default 0.0 (disabled — gamma too uniform).
            lambda_cons  : weight for L_cons. Default 0.0 (disabled — causes collapse).
            lambda_reg   : weight for L_reg (EN consistency). Default 10.0.
                           Paper uses λ_Reg=50 but that paper has 3 LLM passes with
                           much larger LLM loss magnitude. For encoder-only with QA loss,
                           start at 10.0 and tune upward if EN EM still drops.
        """
        super().__init__()
        self.lambda_ot    = lambda_ot
        self.lambda_span  = lambda_span
        self.lambda_cons  = lambda_cons
        self.lambda_reg   = lambda_reg
```

### 2b. Sửa `forward`

**Tìm:**
```python
    def forward(
        self,
        L_ot: torch.Tensor,
        L_span: torch.Tensor,
        L_cons: torch.Tensor,
        epoch: int,
    ) -> dict[str, torch.Tensor]:
        """
        Combine loss components with epoch-based curriculum weighting.

        Returns:
            dict with "total", "ot", "span", "cons", "cons_weight", "span_weight"
        """
        if epoch == 1:
            w_cons = 0.0
            w_span = 0.0
        elif epoch in [2, 3]:
            w_cons = 1.0
            w_span = 0.0
        else:
            w_cons = 1.0
            w_span = 1.0

        L_total = (
            self.lambda_ot * L_ot
            + self.lambda_span * w_span * L_span
            + self.lambda_cons * w_cons * L_cons
        )

        return {
            "total":        L_total,
            "ot":           L_ot.detach(),
            "span":         (L_span.detach() * w_span),
            "cons":         (L_cons.detach() * w_cons),
            "cons_weight":  torch.tensor(float(w_cons)),
            "span_weight":  torch.tensor(float(w_span)),
        }
```

**Thay bằng:**
```python
    def forward(
        self,
        L_ot: torch.Tensor,
        L_reg: torch.Tensor,
        L_span: torch.Tensor,
        L_cons: torch.Tensor,
        epoch: int,
    ) -> dict[str, torch.Tensor]:
        """
        Combine loss components.

        L_reg is active from epoch 1 (always on — EN anchor must be protected
        from the very first gradient step).
        L_span and L_cons are disabled by default (lambda=0.0).

        Returns:
            dict with "total", "ot", "reg", "span", "cons"
        """
        L_total = (
            self.lambda_ot   * L_ot
            + self.lambda_reg  * L_reg
            + self.lambda_span * L_span
            + self.lambda_cons * L_cons
        )

        return {
            "total": L_total,
            "ot":    L_ot.detach(),
            "reg":   L_reg.detach(),
            "span":  L_span.detach(),
            "cons":  L_cons.detach(),
        }
```

---

## Thay đổi 3: `train_stage2.py` — Thêm forward pass thứ 3 và wire L_Reg

### 3a. Sửa `stage2_step`

**Tìm toàn bộ function `stage2_step`** (dòng 190–281) và thay bằng:

```python
def stage2_step(
    batch: dict,
    model,
    criterion,
    stage2_loss,
    epsilon: float,
    n_iters: int,
    epoch: int,
    device: torch.device,
) -> dict:
    """
    One Stage 2 training step with THREE forward passes (per paper):
      1. EN branch — LoRA OFF, no_grad  → h_en_frz  (frozen anchor)
      2. EN branch — LoRA ON,  with_grad → h_en_lora (for L_Reg)
      3. VI branch — LoRA ON,  with_grad → h_vi

    Loss = λ_ot * L_ot + λ_reg * L_reg
    L_reg prevents LoRA shared weights from drifting EN representations.

    Returns:
        dict with all loss tensors and debug info
    """
    from phase3_loss.losses import (
        sinkhorn_masked, compute_span_loss, compute_cons_loss,
        compute_reg_loss, gamma_entropy, _extract_question_embeddings,
    )

    # ── 1. EN branch — LoRA OFF, no gradient (frozen anchor) ────
    with torch.no_grad():
        with model.backbone.disable_adapter():
            en_frz_out = model(batch, branch="en")
            h_en_frz   = en_frz_out["hidden"]        # (B, T_en, H) — detached anchor
            en_mask    = ~en_frz_out["en_pad_mask"]  # (B, T_en) True = real token

            # QA head on frozen EN → pseudo-label logits (for L_span if ever used)
            en_q_emb, en_q_mask = _extract_question_embeddings(
                h_en_frz, batch["en_question_end"]
            )
            en_start_logits, en_end_logits, _ = criterion.qa_head(
                h_en_frz, en_q_emb, en_q_mask
            )

    # ── 2. EN branch — LoRA ON, with gradient (for L_Reg) ───────
    # NOTE: LoRA is ON here — gradient flows through LoRA weights via L_Reg
    en_lora_out = model(batch, branch="en")
    h_en_lora   = en_lora_out["hidden"]   # (B, T_en, H) — has gradient

    # ── 3. VI branch — LoRA ON, with gradient ───────────────────
    vi_out    = model(batch, branch="vi")
    h_vi      = vi_out["hidden"]           # (B, T_vi, H)
    vi_mask   = ~vi_out["vi_pad_mask"]     # (B, T_vi) True = real token

    vi_q_emb, vi_q_mask = _extract_question_embeddings(
        h_vi, batch["vi_question_end"]
    )
    vi_start_logits, vi_end_logits, _ = criterion.qa_head(
        h_vi, vi_q_emb, vi_q_mask
    )

    # ── 4. Sinkhorn OT (uses frozen h_en_frz as anchor) ─────────
    gamma_list, L_ot = sinkhorn_masked(
        h_en_frz, h_vi, en_mask, vi_mask,
        epsilon=epsilon, n_iters=n_iters,
    )

    # ── 5. EN Consistency Regularisation (KEY NEW LOSS) ─────────
    L_reg = compute_reg_loss(h_en_lora, h_en_frz, en_mask)

    # ── 6. Span and Cons losses (disabled by default, kept for ablation) ──
    L_span = compute_span_loss(
        gamma_list, en_start_logits, en_end_logits,
        vi_start_logits, vi_end_logits,
        en_mask, vi_mask,
    )
    L_cons = compute_cons_loss(gamma_list, h_en_frz, h_vi, en_mask, vi_mask)

    # ── 7. Combine with curriculum ───────────────────────────────
    losses = stage2_loss(L_ot, L_reg, L_span, L_cons, epoch)

    # ── 8. Debug metrics ─────────────────────────────────────────
    with torch.no_grad():
        g_entropy = gamma_entropy(gamma_list)
        import math
        avg_n_en = en_mask.sum(dim=1).float().mean().item()
        avg_n_vi = vi_mask.sum(dim=1).float().mean().item()
        h_max    = math.log(max(avg_n_en * avg_n_vi, 1.0))
        h_ratio  = g_entropy / h_max if h_max > 0 else 0

        if h_ratio > 0.90:
            log.warning(
                f"  [Gamma] entropy ratio={h_ratio:.2f} "
                f"(H={g_entropy:.2f}/H_max={h_max:.2f}) — near uniform"
            )
        elif h_ratio < 0.30:
            log.warning(f"  [Gamma] entropy ratio={h_ratio:.2f} — may be collapsed")
        else:
            log.info(f"  [Gamma] entropy ratio={h_ratio:.2f} H={g_entropy:.2f} — healthy")

    losses["gamma_entropy"] = g_entropy
    return losses
```

### 3b. Sửa `STAGE2_CONFIG`

**Tìm:**
```python
    # Loss weights
    "lambda_ot"       : 1.0,
    "lambda_span"     : 1.0,
    "lambda_cons"     : 0.5,

    # OT hyperparameters
    "epsilon"         : 0.05,       # Sinkhorn regularization
```

**Thay bằng:**
```python
    # Loss weights
    "lambda_ot"       : 1.0,
    "lambda_reg"      : 10.0,   # EN consistency regularisation (paper: 50, start lower for encoder)
    "lambda_span"     : 0.0,    # Disabled — gamma too uniform for reliable pseudo-labels
    "lambda_cons"     : 0.0,    # Disabled — causes collapse with uniform gamma

    # OT hyperparameters
    "epsilon"         : 0.1,    # Restored to paper default (0.05 hurts XSQuAD per ablation)
```

### 3c. Sửa `Stage2Loss` instantiation trong `run_stage2`

**Tìm:**
```python
    stage2_loss = Stage2Loss(
        lambda_ot   = config["lambda_ot"],
        lambda_span = config["lambda_span"],
        lambda_cons = config["lambda_cons"],
    ).to(device)
```

**Thay bằng:**
```python
    stage2_loss = Stage2Loss(
        lambda_ot   = config["lambda_ot"],
        lambda_reg  = config["lambda_reg"],
        lambda_span = config["lambda_span"],
        lambda_cons = config["lambda_cons"],
    ).to(device)
```

### 3d. Thêm `--lambda_reg` vào `parse_args`

**Tìm:**
```python
    parser.add_argument("--lambda_ot",      type=float, default=STAGE2_CONFIG["lambda_ot"])
    parser.add_argument("--lambda_span",    type=float, default=STAGE2_CONFIG["lambda_span"])
    parser.add_argument("--lambda_cons",    type=float, default=STAGE2_CONFIG["lambda_cons"])
```

**Thay bằng:**
```python
    parser.add_argument("--lambda_ot",      type=float, default=STAGE2_CONFIG["lambda_ot"])
    parser.add_argument("--lambda_reg",     type=float, default=STAGE2_CONFIG["lambda_reg"])
    parser.add_argument("--lambda_span",    type=float, default=STAGE2_CONFIG["lambda_span"])
    parser.add_argument("--lambda_cons",    type=float, default=STAGE2_CONFIG["lambda_cons"])
```

### 3e. Sửa logging trong training loop

**Tìm:**
```python
                log.info(
                    f"  Step {global_step} | "
                    f"total={losses['total'].item():.4f} | "
                    f"ot={losses['ot'].item():.4f} | "
                    f"span={losses['span'].item():.4f} | "
                    f"cons={losses['cons'].item():.4f} | "
                    f"w_cons={w_cons_val:.3f} | "
                    f"γ_H={g_ent:.2f}"
                )
```

**Thay bằng:**
```python
                log.info(
                    f"  Step {global_step} | "
                    f"total={losses['total'].item():.4f} | "
                    f"ot={losses['ot'].item():.4f} | "
                    f"reg={losses['reg'].item():.4f} | "
                    f"γ_H={g_ent:.2f}"
                )
```

**Tìm:**
```python
                writer.add_scalar("Loss/Stage2_Total", losses["total"].item(), global_step)
                writer.add_scalar("Loss/OT",           losses["ot"].item(),    global_step)
                writer.add_scalar("Loss/Span",         losses["span"].item(),  global_step)
                writer.add_scalar("Loss/Cons",         losses["cons"].item(),  global_step)
                writer.add_scalar("Lambda/Cons_Weight", w_cons_val,            global_step)
                writer.add_scalar("Debug/Gamma_Entropy", g_ent,                global_step)
```

**Thay bằng:**
```python
                writer.add_scalar("Loss/Stage2_Total",  losses["total"].item(), global_step)
                writer.add_scalar("Loss/OT",            losses["ot"].item(),    global_step)
                writer.add_scalar("Loss/Reg",           losses["reg"].item(),   global_step)
                writer.add_scalar("Debug/Gamma_Entropy", g_ent,                 global_step)
```

**Tìm** (epoch summary logging):
```python
        avg = {k: v / max(step_count, 1) for k, v in epoch_losses.items()}
        log.info(
            f"Epoch {epoch} avg | total={avg['total']:.4f} | "
            f"ot={avg['ot']:.4f} | span={avg['span']:.4f} | cons={avg['cons']:.4f}"
        )
```

**Thay bằng:**
```python
        avg = {k: v / max(step_count, 1) for k, v in epoch_losses.items()}
        log.info(
            f"Epoch {epoch} avg | total={avg['total']:.4f} | "
            f"ot={avg['ot']:.4f} | reg={avg['reg']:.4f}"
        )
```

**Tìm** (epoch_losses init):
```python
        epoch_losses = {"total": 0.0, "ot": 0.0, "span": 0.0, "cons": 0.0}
```

**Thay bằng:**
```python
        epoch_losses = {"total": 0.0, "ot": 0.0, "reg": 0.0}
```

**Tìm** (accumulation loop):
```python
            for k in ("total", "ot", "span", "cons"):
```

**Thay bằng:**
```python
            for k in ("total", "ot", "reg"):
```

---

## Checklist xác nhận (lightweight, không cần GPU)

```bash
# 1. compute_reg_loss có trong losses.py không
grep -n "def compute_reg_loss" losses.py

# 2. Stage2Loss.forward nhận L_reg không
grep -n "L_reg" losses.py

# 3. stage2_step có 3 forward pass không
grep -n "disable_adapter\|branch=\"en\"\|branch=\"vi\"" train_stage2.py

# 4. lambda_reg có trong STAGE2_CONFIG không
grep -n "lambda_reg" train_stage2.py

# 5. epsilon về 0.1
grep -n "epsilon" train_stage2.py | head -5
```

---

## Không được thay đổi

- `sinkhorn_masked()` — không đổi
- `compute_cons_loss()` — giữ nguyên để dùng cho ablation sau
- `compute_span_loss()` — giữ nguyên để dùng cho ablation sau
- `quick_eval.py` — không đổi
- `model_core.py` — `branch="en"` hiện tại đã đúng, không cần thêm branch mới
- Stage 1 code trong `losses.py` (dòng 1–680) — không đổi
- `save_stage2_checkpoint()` và HuggingFace upload blocks — không đổi

---

## Lưu ý về `lambda_reg`

Paper dùng `λ_Reg=50` nhưng đó là với Llama-3-8B có LLM loss rất lớn (~3–8).
Với encoder XLM-R, QA loss nhỏ hơn nhiều (~0.2–0.5), nên bắt đầu với `lambda_reg=10`.

Nếu sau 2 epoch EN EM vẫn drop > 5 pts → tăng lên 20–30.  
Nếu VI EM không cải thiện sau 3 epoch → giảm xuống 5.