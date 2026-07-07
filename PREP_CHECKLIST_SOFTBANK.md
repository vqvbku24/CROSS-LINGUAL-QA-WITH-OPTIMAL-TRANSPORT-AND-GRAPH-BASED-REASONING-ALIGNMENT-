# Checklist chuẩn bị SoftBank HPC — từ Docker build đến HF download/upload

Giả định: image dùng cho project cross-lingual QA (XLM-R + LoRA + Optimal Transport),
nên Dockerfile bên dưới có sẵn `transformers`, `peft`, `datasets`, `accelerate`,
`huggingface_hub`, `POT` (Sinkhorn OT), `sentencepiece`, `evaluate`.
Nếu thiếu lib nào, chỉ cần thêm dòng vào `requirements.txt` ở Bước 2 rồi build lại — không cần đổi gì khác.

Nguyên tắc quan trọng: **không bake model weights / dataset vào image**. Container chỉ chứa
Python + library. Weights/dataset tải về `/lustre` và mount vào, giống code — vì image sẽ
rất nặng và mỗi lần đổi checkpoint lại phải rebuild + scp lại cả GB.

---

## BƯỚC 1 — Chuẩn bị token Hugging Face (làm trước, không cần server)

1. Lấy token tại https://huggingface.co/settings/tokens
   - Nếu cần **download** model/dataset private → token `read`
   - Nếu cần **upload** model lên HF Hub → token `write`
2. Lưu token vào 1 file local, KHÔNG commit vào git, KHÔNG bake vào Docker image:
   ```bash
   mkdir -p ~/.hf_secrets
   echo "hf_xxxxxxxxxxxxxxxxxxxxx" > ~/.hf_secrets/token
   chmod 600 ~/.hf_secrets/token
   ```
3. Token sẽ được copy lên server ở Bước 6 (không copy qua Docker image).

---

## BƯỚC 2 — Viết Dockerfile + requirements.txt (local)

```bash
mkdir -p ~/pytorch-simple-image
cd ~/pytorch-simple-image
```

`requirements.txt`:

```text
transformers==4.44.2
peft==0.12.0
datasets==2.20.0
accelerate==0.33.0
huggingface_hub==0.24.6
evaluate==0.4.2
sentencepiece==0.2.0
POT==0.9.4
scikit-learn==1.5.1
tqdm
```

`Dockerfile`:

```dockerfile
FROM pytorch/pytorch:2.12.1-cuda12.6-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    git vim nano wget curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

CMD ["bash"]
```

> Lưu ý: nếu bạn cần một version PyTorch/CUDA khác so với base image trên, nói mình biết
> để đổi base image trước khi build, tránh phải build lại từ đầu.

---

## BƯỚC 3 — Build image (local)

Mac/Windows (ép platform amd64 vì server là Linux x86_64):

```bash
docker buildx build --platform linux/amd64 -t comer-pytorch:cuda .
```

Linux x86_64:

```bash
docker build -t comer-pytorch:cuda .
```

## BƯỚC 4 — Test nhanh local

```bash
docker run --rm comer-pytorch:cuda \
  python -c "import torch, transformers, peft, datasets, ot; \
  print('torch', torch.__version__); \
  print('transformers', transformers.__version__); \
  print('peft', peft.__version__); \
  print('ot(POT) OK')"
```

Trên Mac thì `torch.cuda.is_available() = False` là bình thường, không phải lỗi.

---

## BƯỚC 5 — Save image thành `.tar.gz`

```bash
docker save comer-pytorch:cuda | gzip > comer-pytorch-cuda.tar.gz
ls -lh comer-pytorch-cuda.tar.gz
```

---

## BƯỚC 6 — scp image + token HF lên server

Mở tunnel (giữ terminal này chạy nền):

```bash
ssh -N softbank-bastion
```

Terminal khác — tạo thư mục rồi copy:

```bash
ssh -p 2222 user129002@localhost \
  "mkdir -p /lustre/user129002/Research_infra/images /lustre/user129002/.hf_secrets"

scp -P 2222 comer-pytorch-cuda.tar.gz \
  user129002@localhost:/lustre/user129002/Research_infra/images/

scp -P 2222 ~/.hf_secrets/token \
  user129002@localhost:/lustre/user129002/.hf_secrets/token
```

Với tốc độ ~10MB/s như bạn đã đo, file vài GB sẽ mất vài phút — canh thời gian trước khi cần dùng.

---

## BƯỚC 7 — Trên server: giải nén + convert `.sqsh`

```bash
ssh -p 2222 user129002@localhost
cd /lustre/user129002/Research_infra/images

gunzip comer-pytorch-cuda.tar.gz

enroot import -o comer-pytorch-cuda.sqsh \
  docker-archive://./comer-pytorch-cuda.tar
```

Nếu lỗi `docker-archive://` không hỗ trợ:

```bash
enroot import -o comer-pytorch-cuda.sqsh ./comer-pytorch-cuda.tar
```

---

## BƯỚC 8 — Set enroot path cố định (thêm vào `~/.bashrc` trên server, làm 1 lần)

```bash
cat >> ~/.bashrc <<'EOF'

export ENROOT_CACHE_PATH=/lustre/user129002/.cache/enroot
export ENROOT_DATA_PATH=/lustre/user129002/.local/share/enroot
export HF_HOME=/lustre/user129002/.cache/huggingface
export HF_TOKEN=$(cat /lustre/user129002/.hf_secrets/token 2>/dev/null)
EOF

source ~/.bashrc

mkdir -p /lustre/user129002/.cache/enroot \
         /lustre/user129002/.local/share/enroot \
         /lustre/user129002/.cache/huggingface
```

`HF_HOME` trỏ vào `/lustre` để cache model/dataset tải từ HF được lưu bền, không mất khi
container/job kết thúc, và mount thẳng vào container ở Bước 10.

---

## BƯỚC 9 — Tạo enroot container

```bash
cd /lustre/user129002/Research_infra/images

enroot create --name comer-pytorch comer-pytorch-cuda.sqsh
enroot list   # kỳ vọng thấy "comer-pytorch"
```

---

## BƯỚC 10 — Xin GPU và start container, mount cả code lẫn HF cache

```bash
srun -p 129-partition --gres=gpu:1 --pty bash
```

Trong shell GPU node:

```bash
export ENROOT_CACHE_PATH=/lustre/user129002/.cache/enroot
export ENROOT_DATA_PATH=/lustre/user129002/.local/share/enroot

enroot start \
  --mount /lustre/user129002/Research_infra/CoMER/CoMER-fisher:/workspace/CoMER-fisher \
  --mount /lustre/user129002/.cache/huggingface:/root/.cache/huggingface \
  --env HF_TOKEN \
  comer-pytorch \
  bash
```

---

## BƯỚC 11 — Trong container: login HF + test download/upload

```bash
cd /workspace/CoMER-fisher

python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# login HF bằng token đã mount qua biến môi trường
huggingface-cli login --token $HF_TOKEN

# test download model/dataset
python -c "
from transformers import AutoModel, AutoTokenizer
m = AutoModel.from_pretrained('xlm-roberta-base')
t = AutoTokenizer.from_pretrained('xlm-roberta-base')
print('download OK')
"

# test upload (ví dụ checkpoint sau khi train xong)
python -c "
from huggingface_hub import HfApi
api = HfApi()
# api.upload_folder(folder_path='./checkpoints/run1', repo_id='<your-username>/<repo-name>', repo_type='model')
print('HfApi sẵn sàng, bỏ comment dòng upload_folder khi có checkpoint thật')
"
```

> Vì `HF_HOME` đã mount vào `/lustre`, model/dataset tải về 1 lần sẽ được cache lại — lần
> sau chạy job khác không cần tải lại từ HF Hub nữa, chỉ đọc từ cache local.

---

## BƯỚC 12 — VS Code Remote-SSH để sửa code trực tiếp (không cần rsync lại)

`~/.ssh/config` local:

```sshconfig
Host softbank-bastion
    HostName 211.111.218.41
    User user129002
    Port 50002
    IdentityFile ~/.ssh/id_rsa_quapp
    IdentitiesOnly yes

Host softbank-work
    HostName 10.120.58.68
    Port 22
    User user129002
    IdentityFile ~/.ssh/id_rsa_quapp
    IdentitiesOnly yes
    ProxyJump softbank-bastion
    ServerAliveInterval 60
```

Trong VS Code: `Remote-SSH: Connect to Host` → `softbank-work` → mở folder
`/lustre/user129002/Research_infra/CoMER/CoMER-fisher`.

Script tiện dùng, gộp xin GPU + start container thành 1 lệnh (chạy trong terminal VS Code
trên server):

```bash
cat > ~/run_gpu.sh <<'EOF'
#!/bin/bash
export ENROOT_CACHE_PATH=/lustre/user129002/.cache/enroot
export ENROOT_DATA_PATH=/lustre/user129002/.local/share/enroot
export HF_HOME=/lustre/user129002/.cache/huggingface
export HF_TOKEN=$(cat /lustre/user129002/.hf_secrets/token 2>/dev/null)

srun -p 129-partition --gres=gpu:1 --pty \
  enroot start \
    --mount /lustre/user129002/Research_infra/CoMER/CoMER-fisher:/workspace/CoMER-fisher \
    --mount /lustre/user129002/.cache/huggingface:/root/.cache/huggingface \
    --env HF_TOKEN \
    comer-pytorch \
    bash -lc "cd /workspace/CoMER-fisher && bash"
EOF
chmod +x ~/run_gpu.sh
```

Tối chỉ cần: mở VS Code Remote → terminal gõ `./run_gpu.sh` → có GPU + container +
code mới nhất + HF cache, tất cả trong 1 lệnh.

---

## Checklist tổng — làm trước khi tối vào dùng

```text
[ ] Có HF token (read/write tùy nhu cầu), lưu ở ~/.hf_secrets/token local
[ ] requirements.txt liệt kê đủ lib cần (transformers, peft, datasets, POT, ...)
[ ] docker build thành công local
[ ] docker run test import các lib thành công
[ ] docker save + gzip xong file .tar.gz
[ ] scp .tar.gz + token HF lên /lustre (canh thời gian theo tốc độ ~10MB/s)
[ ] Trên server: gunzip, enroot import thành .sqsh thành công
[ ] Đã thêm ENROOT_*/HF_HOME/HF_TOKEN vào ~/.bashrc server
[ ] enroot create --name comer-pytorch thành công, enroot list thấy tên
[ ] VS Code Remote-SSH connect vào softbank-work OK, mở đúng folder code
[ ] run_gpu.sh tạo xong, chmod +x
[ ] Tối: srun xin GPU OK → enroot start mount code + HF cache OK
[ ] Trong container: huggingface-cli login bằng $HF_TOKEN OK
[ ] Test download 1 model nhỏ (xlm-roberta-base) thành công
[ ] Test HfApi() sẵn sàng để upload checkpoint khi cần
```
