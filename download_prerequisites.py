#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
download_prerequisites.py

Kiểm tra và tự động tải các file tiền đề quan trọng từ Hugging Face Hub:
1. checkpoints/stage1_squad_best.pt (~1.1 GB)
2. dataset/IndicSQuAD/train_hindi.json (~285 MB)

Đảm bảo dữ liệu và checkpoint gốc sẵn sàng đúng vị trí trước khi tiến hành huấn luyện.
"""

import os
import sys
import shutil

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(BASE_DIR, ".env"))
except ImportError:
    pass

DEFAULT_HF_REPO = "vinhvo1205/Sinkhorn_2_stages"


def get_hf_token() -> str | None:
    token = os.environ.get("HF_TOKEN", None)
    if not token:
        token_file = os.path.join(BASE_DIR, ".hf_token")
        if os.path.exists(token_file):
            with open(token_file, "r", encoding="utf-8") as f:
                token = f.read().strip()
    return token


def check_and_download_file(
    repo_id: str,
    path_in_repo: str,
    local_path: str,
    min_size_bytes: int = 1024 * 1024,
) -> bool:
    """
    Kiểm tra file local. Nếu chưa có hoặc kích thước nhỏ hơn min_size_bytes thì tải từ HF.
    """
    if os.path.exists(local_path):
        current_size = os.path.getsize(local_path)
        if current_size >= min_size_bytes:
            print(f"  ✓ Đã có sẵn: {local_path} ({current_size / (1024*1024):.1f} MB)")
            return True
        else:
            print(f"  ⚠ File {local_path} không hoàn chỉnh ({current_size} bytes). Tiến hành tải lại...")

    print(f"  ⬇ Đang tải từ Hugging Face: {repo_id} -> {path_in_repo}...")
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    try:
        from huggingface_hub import hf_hub_download

        token = get_hf_token()
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=path_in_repo,
            repo_type="model",
            token=token,
        )
        shutil.copy(downloaded, local_path)
        final_size = os.path.getsize(local_path)
        print(f"  ✅ Tải thành công và lưu vào: {local_path} ({final_size / (1024*1024):.1f} MB)")
        return True
    except Exception as e:
        print(f"  ❌ Lỗi khi tải {path_in_repo} từ Hugging Face: {e}", file=sys.stderr)
        return False


def main():
    repo_id = os.environ.get("HF_REPO_ID", DEFAULT_HF_REPO)
    print("=" * 70)
    print("  KIỂM TRA & TẢI PREREQUISITES TỪ HUGGING FACE HUB")
    print(f"  Repository: {repo_id}")
    print("=" * 70)

    prerequisites = [
        (
            "checkpoints/stage1_squad_best.pt",
            os.path.join(BASE_DIR, "checkpoints", "stage1_squad_best.pt"),
            100 * 1024 * 1024,  # Stage 1 pt ~ 1.1GB, tối thiểu phải > 100MB
        ),
        (
            "dataset/IndicSQuAD/train_hindi.json",
            os.path.join(BASE_DIR, "dataset", "IndicSQuAD", "train_hindi.json"),
            50 * 1024 * 1024,   # Hindi json ~ 285MB, tối thiểu phải > 50MB
        ),
    ]

    all_passed = True
    for path_in_repo, local_path, min_size in prerequisites:
        ok = check_and_download_file(repo_id, path_in_repo, local_path, min_size)
        if not ok:
            all_passed = False

    if not all_passed:
        print("\n❌ CẢNH BÁO: Một số file tiền đề chưa được tải thành công!")
        print("Vui lòng kiểm tra lại kết nối mạng hoặc cấu hình HF_TOKEN trong .env!")
        sys.exit(1)
    else:
        print("\n✨ Tất cả prerequisites đã sẵn sàng đúng vị trí để huấn luyện!")
        sys.exit(0)


if __name__ == "__main__":
    main()
