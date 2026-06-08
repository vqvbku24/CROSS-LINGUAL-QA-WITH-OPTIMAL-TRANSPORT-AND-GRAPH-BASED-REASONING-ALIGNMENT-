# PARAM_TUNING_SPEC.md
> Agent: review từng item, quyết định có apply hay không dựa trên context thực tế (sequence length, GPU memory, loss scale quan sát được khi train).

---

## Priority Table

| Priority | File | Tham số | Hiện tại | Đề xuất | Rủi ro nếu không đổi |
|---|---|---|---|---|---|
| 🔴 HIGH | `losses.py` | `sinkhorn_epsilon` | `0.05` | `0.1` | Span extraction collapse (Arabic −14 F1 theo ablation bài báo ACL) |
| 🟡 MEDIUM | `losses.py` + `train.py` | `sinkhorn_iters` | `50` | `100` | Under-converged transport gradient, alignment signal yếu |
| 🟡 MEDIUM | `train.py` | `lambda_ot` | `0.1` | Scale-check | OT contribution quá nhỏ so với L_qa, không có tác dụng align |

---

## Item 1 — `sinkhorn_epsilon`: 0.05 → 0.1

**File:** `losses.py`
**Function:** `OTAlignmentLoss.__init__()` và `DEFAULT_CONFIG` trong `train.py`

**Problem snippet:**
```python
# losses.py - OTAlignmentLoss.__init__
sinkhorn_epsilon: float = 0.05,  # entropic regularization

# train.py - DEFAULT_CONFIG
"sinkhorn_epsilon"  : 0.05,
```

**Lý do:**
Bài báo ACL ablate ε ∈ {0.05, 0.1, 0.2} trên cùng task extractive QA (XSQuAD):
- ε=0.05 → plan quá **sharp**, cưỡng bức one-to-one token alignment → XSQuAD F1 sập từ 67.6 xuống 63.1, Arabic riêng giảm ~14 F1
- ε=0.1 → balance tốt nhất giữa alignment signal và generative capacity
- Span extraction cần plan **mềm**: một EN span (nhiều token) phải phân phối mass sang nhiều VI token, không bị ép về một điểm

**Fix snippet:**
```python
# losses.py - OTAlignmentLoss.__init__
sinkhorn_epsilon: float = 0.1,   # ← đổi từ 0.05

# train.py - DEFAULT_CONFIG
"sinkhorn_epsilon"  : 0.1,       # ← đổi từ 0.05
```

**No-touch zones:** Logic Sinkhorn solver, PAD masking, marginal computation — không đổi gì ngoài giá trị default.

---

## Item 2 — `sinkhorn_iters`: 50 → 100

**File:** `losses.py` + `train.py`
**Function:** `OTAlignmentLoss.__init__()` và `DEFAULT_CONFIG`

**Problem snippet:**
```python
# losses.py
sinkhorn_iters: int = 50,

# train.py
"sinkhorn_iters"    : 50,
```

**Lý do:**
Bài báo cho thấy K=50 là "under-converged" → gradient từ transport plan nhiễu, làm yếu alignment pressure. K=100 là điểm tối ưu efficiency–quality.

**Lưu ý cho agent:** Với dynamic truncation (T_en × T_vi << 512²), wall-clock tăng khi đổi K=50→100 sẽ **nhỏ hơn nhiều** so với bài báo (họ tăng từ 23.98h → 34.01h trên full 512 tokens). Agent nên đo thực tế time/batch trước khi quyết định.

**Fix snippet:**
```python
# losses.py
sinkhorn_iters: int = 100,       # ← đổi từ 50

# train.py
"sinkhorn_iters"    : 100,       # ← đổi từ 50
```

**No-touch zones:** Vòng lặp Sinkhorn, log-domain stabilization — không đổi.

---

## Item 3 — Scale-check `lambda_ot`

**File:** `train.py`
**Function:** `DEFAULT_CONFIG`, `run_training()`

**Problem snippet:**
```python
"lambda_ot"         : 0.1,
```

**Lý do:**
Bài báo dùng nguyên tắc: `λ_OT * L_OT` nên đóng góp ~20–40% của `L_qa` để có tác dụng thực sự. Với `lambda_ot=0.1`, nếu L_ot ~0.3 thì contribution chỉ ~0.03, trong khi L_qa ~2–3 ở epoch đầu → OT gần như không có gradient ảnh hưởng.

**Không phải fix cứng** — agent cần quan sát TensorBoard:
- Check `Loss/OT (Transport)` vs `Loss/QA` ở bước đầu training
- Nếu `lambda_ot * L_ot < 5% của L_qa` → tăng `lambda_ot` lên 0.3–0.5
- Nếu đã ~20–40% → giữ nguyên

**Fix snippet (conditional):**
```python
# Chỉ apply nếu quan sát thấy OT contribution < 5% L_qa
"lambda_ot"         : 0.3,   # ← tăng từ 0.1, nếu cần
```

**No-touch zones:** Curriculum warmup schedule cho lambda_ot trong `run_training()` — logic annealing không đổi, chỉ đổi target value.