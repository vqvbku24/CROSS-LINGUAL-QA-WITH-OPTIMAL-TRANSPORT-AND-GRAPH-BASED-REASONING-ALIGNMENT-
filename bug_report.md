# BUGFIX SPEC — requirements.txt thiếu dependency trước khi build Docker

## Priority Table

| # | Mức độ | Vấn đề | File |
|---|---|---|---|
| 1 | CRITICAL | Thiếu `peft` — code import trực tiếp, container sẽ crash khi chạy model | `requirements.txt` |
| 2 | IMPORTANT | Thiếu `accelerate` — cần cho multi-GPU / device_map trên HPC node 8-GPU | `requirements.txt` |
| 3 | IMPORTANT | Thiếu `scikit-learn` — cần nếu có tính metric (F1/EM) hoặc dùng gián tiếp qua `evaluate` | `requirements.txt` |
| 4 | OPTIONAL | Thiếu `evaluate` — chỉ thêm nếu thực sự dùng HF `evaluate` lib để tính metric | `requirements.txt` |
| 5 | OPTIONAL | `sentencepiece` chưa pin version | `requirements.txt` |

---

## 1. [CRITICAL] Thiếu `peft`

**File & vị trí:** `phase2_model/model_core.py:19`

**Snippet vấn đề (code có nhưng requirements.txt không khai báo):**
```python
from peft import get_peft_model, LoraConfig, TaskType
```

**Fix — thêm vào `requirements.txt`:**
```diff
 transformers==4.41.2
+peft==0.12.0
 datasets==3.2.0
```

> Chọn `peft==0.12.0` vì tương thích với `transformers==4.41.2` (peft 0.12.x hỗ trợ transformers >=4.31, <4.45 không vấn đề). Không đổi version `transformers` hiện tại.

---

## 2. [IMPORTANT] Thiếu `accelerate`

**Vấn đề:** Không thấy import trực tiếp trong code hiện tại, nhưng chạy multi-GPU trên node `fcdgx00090` (8 GPU) qua `Trainer`/`device_map="auto"` sau này sẽ cần. Thêm trước để không phải rebuild image giữa chừng khi mở rộng sang multi-GPU.

**Fix:**
```diff
 huggingface_hub==0.23.0
+accelerate==0.33.0
 sentencepiece==0.2.0
```

---

## 3. [IMPORTANT] Thiếu `scikit-learn`

**Vấn đề:** Cần xác nhận có dùng cho tính metric (ví dụ cosine similarity, clustering trong OT diagnostics) hay không trước khi thêm.

**Fix (nếu xác nhận cần):**
```diff
+scikit-learn==1.5.1
```

**Điều kiện:** Chỉ thêm sau khi grep xác nhận có dùng `from sklearn import ...` ở đâu đó trong repo:
```bash
grep -r "sklearn" --include="*.py" .
```
Nếu không có kết quả nào → bỏ qua bước này, không thêm lib không dùng.

---

## 4. [OPTIONAL] `evaluate`

Chỉ thêm nếu grep thấy:
```bash
grep -r "import evaluate" --include="*.py" .
```
Nếu không có → không cần, tránh phình requirements không cần thiết.

---

## 5. [OPTIONAL] Pin `sentencepiece`

```diff
-sentencepiece
+sentencepiece==0.2.0
```

---

## No-touch zones

- **KHÔNG đổi** version `transformers` (giữ `4.41.2`), `datasets` (giữ `3.2.0`), `huggingface_hub` (giữ `0.23.0`), `POT` (giữ `0.9.3`) — đây là các version đã chạy thật với code, đổi theo số trong checklist ban đầu (4.44.2/2.20.0/...) là không cần thiết và có rủi ro breaking change.
- **KHÔNG** thêm `evaluate`/`scikit-learn` nếu không grep thấy import thực tế — tránh bloat image và tăng thời gian build/scp không cần thiết.
- **KHÔNG** sửa `.gitignore`, `run.sh`, hay token-loading logic — các phần này đã đúng theo review.

---

## Sau khi fix — rebuild lại từ Bước 3 trong PREP_CHECKLIST_SOFTBANK.md

```bash
cd ~/pytorch-simple-image
docker buildx build --platform linux/amd64 -t comer-pytorch:cuda .
docker run --rm comer-pytorch:cuda \
  python -c "import torch, transformers, peft, datasets, accelerate; print('all imports OK')"
```