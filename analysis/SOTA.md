# Spec: Triển khai 2 SOTA Baseline cho So sánh Cross-Lingual XQA

**Mục tiêu:** Thêm 2 baseline SOTA — được **train lại dưới cùng protocol** với framework hiện tại (không phải cite số published) — để so sánh công bằng trong Table 2. Mỗi baseline nằm trong 1 thư mục riêng, tái sử dụng tối đa hạ tầng Stage 1/Stage 2 đã có.

Hai baseline này được chọn vì chúng **chính là 2 công thức đã được cite trong Table 1** của bài báo (hàng "Contrastive" và hàng "OT"), nên việc reproduce chúng biến 2 citation "suông" thành 2 con số thực nghiệm, củng cố trực tiếp luận điểm cốt lõi của bài (space + domain phải phối hợp, một trục riêng lẻ không đủ).

---

## 0. Nguyên tắc chung (áp dụng cho cả 2 baseline)

### 0.1 Cái gì giữ nguyên (reuse, KHÔNG viết lại)
- **Stage 1 teacher checkpoint** (English SQuAD 2.0, XLM-R-base + dual QA heads) — dùng chung checkpoint đã train, không train lại.
- **Data pipeline**: parallel loader (English SQuAD 2.0 + target-language unlabeled QA text), 3-forward-pass structure mỗi step (EN Frozen / EN LoRA / Target LoRA) như Figure 1 mô tả.
- **Backbone + LoRA config**: XLM-R-base, LoRA r=16, α=32, dropout=0.05, target modules = {query, key, value, dense}.
- **Layer-mixing**: trainable softmax aggregation qua layer 6–9 (giữ nguyên cơ chế, không đổi).
- **$\mathcal{L}_{domain}$ (domain-consistency anchor)**: giữ nguyên công thức, λ_reg = 50.0.
- **$\mathcal{L}_{time}$ (boundary regularization)**: giữ nguyên công thức margin + static schedule (m=1.0 xuyên suốt), theo khuyến nghị M5 đã chọn làm default.
- **$\mathcal{L}_{qa}$**: giữ nguyên, λ_qa = 0.3.
- **Optimizer/schedule**: linear LR scheduler, peak lr = 2×10⁻⁵, warmup ratio 0.1, 6 epochs.
- **Eval harness**: script tính EM/F1 (SQuAD 2.0 style, answerable + unanswerable), Δsrc metric, checkpoint selection script (best joint target-val + trong 1 F1 điểm so với teacher).

### 0.2 Cái gì thay đổi (biến số duy nhất bị cô lập)
Chỉ thay **$\mathcal{L}_{space}$** (hiện là $\lambda_{ot}\mathcal{L}_{ot} + \lambda_{span}\mathcal{L}_{span}$) bằng cơ chế alignment của từng SOTA. Mọi thứ khác giữ nguyên — đúng tinh thần ablation M1–M5 đã làm trong bài.

$$\mathcal{L}_{total}^{baseline} = \lambda_{qa}\mathcal{L}_{qa} + \mathcal{L}_{space}^{baseline} + \mathcal{L}_{domain} + \mathcal{L}_{time}$$

**Quy tắc bắt buộc cho agent:** mỗi file `train_stage2.py` trong 2 thư mục baseline PHẢI import các module dùng chung từ `shared/`, không được copy-paste code — để đảm bảo phần "không đổi" (data, domain anchor, margin, eval) là **byte-for-byte giống nhau** giữa 2 baseline và framework chính. Đây là điều kiện tiên quyết để so sánh có giá trị khoa học.

### 0.4 Quy ước chạy & báo cáo (khớp Table 2/3/4 hiện có)
- Seed chính: **42** (headline number, khớp convention hiện tại của Table 2).
- Ngôn ngữ bắt buộc: **Vietnamese** (khớp scope "in-depth analysis" đã tuyên bố). Arabic/Hindi là optional nếu còn compute.
- Checkpoint selection: dùng lại đúng script hiện có (best joint target-val F1/EM, trong 1 điểm F1 so với teacher).
- Output format `results.json` mỗi baseline:
```json
{
  "config_name": "infoxlm_contrastive_static",
  "seed": 42,
  "language": "vi",
  "epoch_selected": 3,
  "xquad": {"em": 0.0, "f1": 0.0},
  "mlqa": {"em": 0.0, "f1": 0.0},
  "squad_en": {"pre_f1": 72.84, "post_f1": 0.0, "delta_src": 0.0},
  "xquad_en_heldout": {"pre_f1": 75.7, "post_f1": 0.0, "delta_src": 0.0},
  "mlqa_en_heldout": {"pre_f1": 79.8, "post_f1": 0.0, "delta_src": 0.0}
}
```
Format này khớp thẳng với cột trong Table 2/3, agent chỉ cần map trực tiếp khi viết bảng.

---

## 1. BASELINE A: InfoXLM-style Cross-Lingual Contrastive Alignment

**Tham chiếu:** Chi et al., 2021, *InfoXLM: An Information-Theoretic Framework for Cross-Lingual Language Model Pre-training* (NAACL 2021) — cơ chế **Cross-Lingual Contrastive (XLCO)** objective, dùng InfoNCE để kéo gần biểu diễn của các cặp câu song song, đẩy xa các cặp âm (negative) trong batch.

**Vì sao chọn công thức này (không phải bản pretraining-scale gốc):** InfoXLM gốc dùng momentum-contrast queue (~131k negatives) ở **giai đoạn pretraining**. Ở downstream fine-tuning (task của chúng ta), phần lớn các paper reproduce InfoXLM-style contrastive dùng **in-batch negatives**, bỏ momentum queue (queue chỉ cần thiết khi train từ đầu trên corpus khổng lồ). Đây là adaptation chuẩn, đã được nhiều paper downstream áp dụng (vd. CrossAligner, Gritta et al. 2022). Batch của chúng ta vốn đã có sẵn cặp (EN, target) song song mỗi step nhờ parallel loader → không cần queue.

### 1.1 Công thức chính xác

Cho 1 batch B cặp song song (EN, target). Lấy representation đã qua layer-mixing (giống hệt $H^{EN}_{LoRA}$ và $H^{Target}$ trong Figure 1), **mean-pool** qua các token không phải padding (dùng attention mask) để ra vector duy nhất mỗi câu:

$$z^{EN}_i = \text{MeanPool}(H^{EN}_{LoRA,i}), \quad z^{Tgt}_i = \text{MeanPool}(H^{Target}_i)$$

L2-normalize cả hai. Similarity matrix với temperature τ:

$$S_{ij} = \frac{z^{EN}_i \cdot z^{Tgt}_j}{\tau}$$

Symmetric InfoNCE (2 chiều, giống CLIP/XLCO):

$$\mathcal{L}_{en2tgt} = -\frac{1}{B}\sum_{i=1}^{B} \log \frac{\exp(S_{ii})}{\sum_{j=1}^{B}\exp(S_{ij})}$$

$$\mathcal{L}_{tgt2en} = -\frac{1}{B}\sum_{i=1}^{B} \log \frac{\exp(S_{ii})}{\sum_{j=1}^{B}\exp(S_{ji})}$$

$$\mathcal{L}_{contrast} = \frac{1}{2}\left(\mathcal{L}_{en2tgt} + \mathcal{L}_{tgt2en}\right)$$

**Thay thế:**
$$\mathcal{L}_{space}^{InfoXLM} = \lambda_{contrast}\mathcal{L}_{contrast} \quad \text{(bỏ hẳn } \mathcal{L}_{ot}, \mathcal{L}_{span}\text{)}$$

### 1.2 Hyperparameters cần grid search
- **τ (temperature)**: grid {0.05, 0.07, 0.1} — theo convention SimCLR/CLIP/InfoXLM.
- **λ_contrast**: grid {0.1, 0.5, 1.0, 2.0} — InfoNCE loss có scale O(log B), không cần trọng số lớn như λ_reg. Điểm khởi đầu đề xuất: 1.0 (cùng lớp trọng số với λ_span cũ, vì đều là "primary spatial alignment term").

### 1.3 Files cần tạo trong `baselines/infoxlm_contrastive/`

**`loss_contrastive.py`:**
```python
def compute_contrastive_loss(z_en, z_tgt, mask_en, mask_tgt, temperature):
    """
    z_en, z_tgt: (B, T, d) hidden states sau layer-mixing
    mask_en, mask_tgt: (B, T) attention mask (1=valid, 0=pad)
    Trả về: scalar loss L_contrast
    """
    # 1. Mean-pool loại bỏ pad
    # 2. L2-normalize
    # 3. Similarity matrix S = z_en @ z_tgt.T / temperature
    # 4. Symmetric cross-entropy (labels = diagonal, tức arange(B))
    # 5. Return 0.5 * (loss_en2tgt + loss_tgt2en)
    ...
```

**`config.yaml`:**
```yaml
baseline_name: infoxlm_contrastive
temperature: 0.07          # grid: [0.05, 0.07, 0.1]
lambda_contrast: 1.0       # grid: [0.1, 0.5, 1.0, 2.0]
lambda_qa: 0.3             # giữ nguyên
lambda_reg: 50.0           # giữ nguyên (L_domain)
lambda_margin: 1.0         # giữ nguyên, static schedule
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05
epochs: 6
peak_lr: 2e-5
warmup_ratio: 0.1
seed: 42
target_language: vi
```

**`train_stage2.py`:** import `shared/train_loop_base.py`, chỉ override phần tính `L_space` bằng `loss_contrastive.compute_contrastive_loss(...)`. Mọi phần khác (3-forward-pass, `L_qa`, `L_domain`, `L_time`, optimizer, checkpoint selection) gọi thẳng từ `shared/`.

### 1.4 Caveat cần ghi vào README.md / Appendix của bài báo
> "Chúng tôi reproduce InfoXLM's Cross-Lingual Contrastive (XLCO) objective (Chi et al., 2021) dưới dạng in-batch InfoNCE, bỏ momentum-contrast queue vốn chỉ cần thiết ở quy mô pretraining. Đây là adaptation chuẩn cho fine-tuning downstream, nhất quán với các paper reproduce InfoXLM khác (Gritta et al., 2022)."

---

## 2. BASELINE B: MINOTAUR-style OT Posterior Alignment (Sherborne et al., 2023)

**Tham chiếu:** Sherborne, Hosking, Lapata, 2023, *Optimal Transport Posterior Alignment for Cross-lingual Semantic Parsing* (TACL 2023) — phương pháp **MINOTAUR**.

**⚠️ Quan trọng — khác với giả định ban đầu:** Sau khi đọc kỹ bài gốc, MINOTAUR **KHÔNG dùng Sinkhorn transport plan** giữa các token như $\mathcal{L}_{ot}$ hiện tại của bài. Thay vào đó họ dùng:
1. Một mô hình **VAE encoder-decoder** (không phải discriminative span-extraction) với latent variable Z có tham số hoá Gaussian.
2. Alignment 2 tầng: **individual** (giữa các posterior riêng lẻ, dùng công thức **L2-Wasserstein đóng dạng** giữa 2 Gaussian — KHÔNG cần Sinkhorn) + **aggregate** (giữa phân phối biên/marginal, dùng **MMD** với IMQ kernel).

Vì kiến trúc gốc là generative semantic parsing (có decoder sinh logical form), còn bài của chúng ta là discriminative span-extraction (không có decoder), **đây là một adaptation, không phải reproduce y hệt**. Cần ghi rõ điều này trong Appendix, đúng tinh thần caveat đã làm với MAD-X.

### 2.1 Công thức gốc (Sherborne et al., Eq 6, 11, 12, 3-5)

**Tham số hoá posterior** (Eq 6): với sequence T token, encoder cho ra
$$\mathbf{z} = Q_\phi(x) \sim \mathcal{N}(\boldsymbol{\mu}, \sigma^2)$$
trong đó $\boldsymbol{\mu} \in \mathbb{R}^{T \times d}$ (mean **riêng cho từng token**) nhưng $\sigma^2 \in \mathbb{R}^d$ là **MỘT variance dùng chung cho cả sequence** (không phải per-token) — đây là simplification có chủ đích của tác giả gốc để giảm nhiễu.

**Individual alignment** $\mathbb{D}_{Z|X}$ — L2-Wasserstein đóng dạng giữa 2 Gaussian (Eq 12), lấy trung bình qua TẤT CẢ cặp token (i,j), không phải qua 1 assignment tối ưu (không có Sinkhorn ở đây):
$$W_2(\mathbf{p},\mathbf{q}) = \|\mu_p - \mu_q\|_2^2 + \text{Tr}\{\Sigma_p + \Sigma_q - 2(\Sigma_p^{1/2}\Sigma_q\Sigma_p^{1/2})^{1/2}\}$$
$$\mathbb{D}_{Z|X}(\mathbf{z}_{EN}, \mathbf{z}_l) = \frac{1}{|\mathbf{z}_{EN}||\mathbf{z}_l|}\sum_{i,j} W_2(z_{EN,i}, z_{l,j})$$

Vì $\Sigma$ là diagonal và **chung cho cả sequence** (không đổi theo token), phần trace rút gọn còn phép trừ element-wise đơn giản:
$$W_2(\mathbf{p},\mathbf{q}) = \|\mu_p - \mu_q\|_2^2 + \|\sigma_p - \sigma_q\|_2^2$$

**Aggregate alignment** $\mathbb{D}_Z$ — MMD với IMQ kernel (Eq 3-5), ước lượng qua batch B mẫu (dùng pooled/sampled representation mỗi câu làm 1 "mẫu" của phân phối biên):
$$k(p,q) = \sum_{s \in S} \frac{s \cdot C}{s\cdot C + \|p-q\|_2^2}, \quad S = \{0.1, 0.2, 0.5, 1, 2, 5, 10\}$$
$$\text{MMD}_k(\mathbf{p},\mathbf{q}) = \frac{1}{B(B-1)}\sum_{i\neq i'} k(p_i,p_{i'}) + \frac{1}{B(B-1)}\sum_{j\neq j'} k(q_j,q_{j'}) - \frac{2}{B^2}\sum_{i,j}k(p_i,q_j)$$

**Tổng hợp** (Eq 11):
$$\mathbb{D}_{MINOTAUR} = \alpha_P \cdot \text{MMD}(\mathbf{z}_{EN}, \mathbf{z}_{tgt}) + \beta_P \cdot \mathbb{D}_{Z|X}(\mathbf{z}_{EN}, \mathbf{z}_{tgt})$$
với $(\alpha_P, \beta_P) = (0.01, 0.5)$ là giá trị tuned gốc của tác giả (dùng làm điểm khởi đầu, cần re-tune vì domain/scale dữ liệu khác).

Tác giả gốc xác nhận qua ablation (Table 3 của họ): kết hợp W2 + MMD **tốt nhất**, W2 vượt trội hơn KL divergence — nên bài của mình **bắt buộc dùng W2 + MMD**, không dùng KL.

### 2.2 Adaptation cho extractive QA (không có decoder/VAE đầy đủ)

Vì task của chúng ta không sinh sequence (không cần decoder, không cần reparameterization để feed vào decoder), có thể lược bỏ phần generative mà vẫn giữ đúng cơ chế alignment cốt lõi:

1. **μ (mean)**: dùng trực tiếp hidden state đã qua layer-mixing hiện có ($H^{EN}_{LoRA}$, $H^{Target}$) làm $\mu_i$ cho mỗi token — **không cần thêm head mới** cho phần mean.
2. **σ² (variance)**: cần **1 head mới, nhẹ** — 1 linear layer nhận **mean-pooled representation** của cả sequence, xuất ra 1 vector variance dùng chung cho toàn sequence (đúng thiết kế gốc "single variance for all T states"):
```python
class PosteriorVarianceHead(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)
    def forward(self, pooled_repr):
        log_var = self.proj(pooled_repr)
        log_var = torch.clamp(log_var, min=-10, max=10)  # ổn định số học
        return torch.exp(0.5 * log_var)  # trả về sigma, không phải variance
```
   → ~590K tham số mới (d×d cho d=768), rất nhỏ so với 2.65M của LoRA, cần train từ đầu (không phải LoRA-adapted).
3. **Không cần decoder, không cần reparameterization-để-sinh-sequence.** Chỉ cần (μ, σ) để tính W2 (dùng moments trực tiếp — công thức đóng dạng không cần sample) và MMD (cần một "mẫu" — dùng $z = \mu_{pooled} + \sigma \cdot \epsilon$, $\epsilon \sim \mathcal{N}(0, I)$, tức reparameterization 1 dòng, không cần decoder theo sau).
4. **Không có $\mathcal{L}_{span}$ tương đương** trong MINOTAUR gốc — bài baseline này không có cơ chế project pseudo-label span nào, đây chính là điểm khác biệt cần nêu trong bài (MINOTAUR alignment là generic, không task-aware như $\mathcal{L}_{span}$ của framework chính).

**Thay thế:**
$$\mathcal{L}_{space}^{MINOTAUR} = \alpha_P \cdot \text{MMD} + \beta_P \cdot \mathbb{D}_{Z|X} \quad \text{(bỏ hẳn } \mathcal{L}_{ot}, \mathcal{L}_{span}\text{, không dùng Sinkhorn solver)}$$

### 2.3 Hyperparameters cần grid search
- $(\alpha_P, \beta_P)$: bắt đầu từ (0.01, 0.5) như bài gốc, grid xung quanh: α_P ∈ {0.005, 0.01, 0.02}, β_P ∈ {0.25, 0.5, 1.0}.
- Kernel scale C trong MMD: C = 2×d (giả định σ_kernel²=1, vì không có prior N(0,I) cố định để match trong setting của chúng ta — khác bài gốc vốn có prior); ghi rõ đây là điều chỉnh cần thiết.
- **Tần suất áp dụng loss**: bài gốc áp dụng **episodic, mỗi k=20 step** (do few-shot, ít data). Trong setting của chúng ta (data-rich, parallel loader mỗi batch đều có sẵn cặp EN-target), khuyến nghị áp dụng **mỗi step** để nhất quán với cách $\mathcal{L}_{ot}$, $\mathcal{L}_{domain}$, $\mathcal{L}_{time}$ đang được áp dụng trong framework chính — đây là **deviation có chủ đích, cần ghi chú rõ trong README/Appendix**.

### 2.4 Files cần tạo trong `baselines/minotaur_ot_posterior/`

**`posterior_head.py`:** class `PosteriorVarianceHead` như code mẫu ở mục 2.2.

**`loss_ot_posterior.py`:**
```python
def compute_w2_individual(mu_en, mu_tgt, sigma_en, sigma_tgt, mask_en, mask_tgt):
    """
    mu_en: (B, T_en, d), mu_tgt: (B, T_tgt, d) — hidden states hiện có, dùng trực tiếp làm mu
    sigma_en: (B, d), sigma_tgt: (B, d) — output của PosteriorVarianceHead, 1 vector/sequence
    Trả về D_{Z|X}: trung bình W2 qua MỌI cặp token (i,j), có áp dụng mask để bỏ pad
    """
    ...

def compute_mmd_aggregate(z_en_sample, z_tgt_sample, kernel_scales=(0.1,0.2,0.5,1,2,5,10)):
    """
    z_en_sample, z_tgt_sample: (B, d) — 1 mẫu (đã reparameterize) mỗi câu trong batch
    Trả về MMD_k theo Eq 4, dùng IMQ kernel Eq 5
    """
    ...

def compute_minotaur_loss(H_en, H_tgt, mask_en, mask_tgt, posterior_head, alpha_p=0.01, beta_p=0.5):
    # 1. mu_en = H_en, mu_tgt = H_tgt (dùng trực tiếp)
    # 2. pooled_en = mean_pool(H_en, mask_en); pooled_tgt = mean_pool(H_tgt, mask_tgt)
    # 3. sigma_en = posterior_head(pooled_en); sigma_tgt = posterior_head(pooled_tgt)
    # 4. D_ZX = compute_w2_individual(mu_en, mu_tgt, sigma_en, sigma_tgt, mask_en, mask_tgt)
    # 5. z_en_sample = pooled_en + sigma_en * randn_like(pooled_en)
    #    z_tgt_sample = pooled_tgt + sigma_tgt * randn_like(pooled_tgt)
    # 6. D_Z = compute_mmd_aggregate(z_en_sample, z_tgt_sample)
    # 7. return alpha_p * D_Z + beta_p * D_ZX
    ...
```

**`config.yaml`:**
```yaml
baseline_name: minotaur_ot_posterior
alpha_p: 0.01              # grid: [0.005, 0.01, 0.02]  (trọng số MMD/aggregate)
beta_p: 0.5                # grid: [0.25, 0.5, 1.0]      (trọng số W2/individual)
mmd_kernel_scale_c: null   # tính tự động = 2 * hidden_dim tại runtime
apply_every_step: true     # deviation so với bài gốc (episodic k=20) — xem README
lambda_qa: 0.3
lambda_reg: 50.0
lambda_margin: 1.0
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05
epochs: 6
peak_lr: 2e-5
warmup_ratio: 0.1
seed: 42
target_language: vi
```

**`train_stage2.py`:** import `shared/train_loop_base.py`, override `L_space` bằng `compute_minotaur_loss(...)`. Thêm `posterior_head` vào danh sách tham số trainable (train from scratch, KHÔNG dùng LoRA cho phần này vì đây là module mới thêm, không phải adapter của backbone có sẵn).

### 2.5 Caveat bắt buộc ghi vào README.md / Appendix của bài báo
> "MINOTAUR (Sherborne et al., 2023) nguyên bản được thiết kế cho một VAE encoder-decoder sinh logical form (semantic parsing), không phải discriminative span-extraction. Chúng tôi adapt cơ chế alignment cốt lõi của họ — Wasserstein-2 giữa posterior riêng lẻ + MMD giữa posterior biên — sang kiến trúc encoder-only của mình bằng cách: (i) dùng trực tiếp hidden state hiện có làm mean của posterior, (ii) thêm 1 head nhẹ dự đoán variance dùng chung cho cả sequence (đúng thiết kế gốc), (iii) bỏ decoder/reparameterization-để-sinh-sequence vì không cần thiết cho mục tiêu alignment. Chúng tôi cũng áp dụng loss này mỗi training step thay vì episodic mỗi k=20 step như bài gốc, vì setting của chúng tôi có sẵn dữ liệu song song mỗi batch (khác với chế độ few-shot khan hiếm dữ liệu của MINOTAUR gốc). Đây là adaptation trung thực với ý tưởng cốt lõi, không phải reproduce y hệt kiến trúc gốc."

---
Điểm mạnh cốt lõi của chiến lược này. Biến Citation thành Thực nghiệm: Hai hàng "Contrastive" và "OT" ở Table 1 (trước đây chỉ là trích dẫn lý thuyết) sẽ biến thành hai con số thực nghiệm khách quan trong Table 2.  Controlled Ablation chuẩn mực: Mọi yếu tố từ backbone (XLM-R-base), LoRA rank ($r=16$), Stage 1 teacher, data loader, cho đến $\mathcal{L}_{domain}$ ($\lambda_{reg}=50.0$) và static margin đều được giữ cố định. Sự thay đổi duy nhất nằm ở $\mathcal{L}_{space}$.  Củng cố trực tiếp luận điểm chính: Kết quả này sẽ chứng minh rõ ràng rằng: khi chỉ dùng không gian biểu diễn đơn lẻ (Contrastive hoặc OT thuần túy) mà thiếu sự phối hợp giữa Space và Domain, hiệu năng transfer sẽ bị suy giảm.  Lưu ý bắt buộc khi trình bày trong bài báoĐể đảm bảo tính minh bạch tuyệt đối khi nộp bài:Ghi rõ các Adaptation trong Appendix: Đưa đúng hai đoạn Caveat ở mục 1.4 và 2.5 của file SOTA.md vào Appendix.  Khẳng định tính đóng góp: Nêu rõ các baseline này được re-implemented/adapted dưới cùng một protocol thử nghiệm để đảm bảo so sánh "apples-to-apples"

## 3. Checklist thứ tự thực hiện cho agent

1. [ ] Verify Stage 1 teacher checkpoint load được, freeze đúng.
2. [ ] Verify `shared/data_loader.py` cho ra đúng 3 forward pass (EN Frozen, EN LoRA, Target LoRA) với batch có cặp song song.
3. [ ] Implement `baselines/infoxlm_contrastive/loss_contrastive.py`, unit test với batch giả (kiểm tra loss > 0, gradient chảy qua LoRA params).
4. [ ] Implement `baselines/minotaur_ot_posterior/posterior_head.py` + `loss_ot_posterior.py`, unit test riêng từng hàm (W2 closed-form, MMD kernel) trước khi ghép.
5. [ ] Chạy grid search nhỏ (τ, λ_contrast cho A; α_P, β_P cho B) trên 1-2 epoch để chọn hyperparameter tốt nhất trước khi chạy full 6 epoch.
6. [ ] Chạy full Stage 2 (6 epoch, seed 42, Vietnamese) cho cả 2 baseline với hyperparameter đã chọn.
7. [ ] Chạy eval harness, xuất `results.json` theo format ở mục 0.4.
8. [ ] Đối chiếu số của cả 2 baseline với "Ours (Coordinated, static)" và "Static Alignment (OT)" đã có trong Table 2 — merge vào bảng.
9. [ ] Viết README.md mỗi thư mục baseline với đúng đoạn caveat ở mục 1.4 và 2.5.
10. [ ] (Optional, nếu còn compute) Lặp lại seed 43, 44 cho multi-seed validation giống Section 4.5/Appendix I.