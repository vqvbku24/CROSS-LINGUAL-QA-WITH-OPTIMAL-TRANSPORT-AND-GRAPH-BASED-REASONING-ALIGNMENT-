import os
import subprocess
import logging
import random
import numpy as np
import torch
import torch.distributed as dist

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# DDP Setup / Cleanup
# ──────────────────────────────────────────────────────────────

def setup_ddp():
    """
    Initialize DDP process group (NCCL backend).
    Must be called at the start of each DDP process (torchrun sets env vars).
    Returns (local_rank, world_size).
    """
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)

    if local_rank == 0:
        log.info(f"DDP initialized: world_size={world_size}")

    return local_rank, world_size


def cleanup_ddp():
    """Destroy DDP process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    """True if this is rank 0 or DDP is not active."""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_local_rank() -> int:
    """Get local rank from env (set by torchrun). Defaults to 0."""
    return int(os.environ.get("LOCAL_RANK", 0))


def is_ddp_active() -> bool:
    """Check if DDP is active (torchrun sets RANK env var)."""
    return "RANK" in os.environ and dist.is_initialized()


# ──────────────────────────────────────────────────────────────
# GPU Auto-Selection (non-DDP only)
# ──────────────────────────────────────────────────────────────

def auto_select_free_gpus(memory_threshold_mb=1000):
    """
    Tự động tìm và gán CUDA_VISIBLE_DEVICES cho các GPU đang trống.
    Một GPU được coi là 'trống' nếu lượng VRAM đang sử dụng < memory_threshold_mb.
    Hàm này phải được gọi trước khi khởi tạo device trong PyTorch.

    Skipped when DDP is active (torchrun handles GPU assignment).
    """
    # DDP: torchrun handles GPU assignment via LOCAL_RANK
    if "RANK" in os.environ:
        log.info("DDP detected (torchrun). Skipping auto_select_free_gpus.")
        return

    if "CUDA_VISIBLE_DEVICES" in os.environ:
        log.info(f"CUDA_VISIBLE_DEVICES đã được thiết lập sẵn: {os.environ['CUDA_VISIBLE_DEVICES']}")
        return

    try:
        # Gọi nvidia-smi để lấy VRAM đang sử dụng
        smi_output = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'],
            encoding='utf-8'
        )
        used_memory = [int(x.strip()) for x in smi_output.strip().split('\n') if x.strip().isdigit()]
        
        free_gpus = []
        for i, used in enumerate(used_memory):
            if used < memory_threshold_mb:
                free_gpus.append(str(i))
                
        if free_gpus:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(free_gpus)
            log.info(f"Tự động chọn các GPU trống: {free_gpus}")
        else:
            log.warning("Không tìm thấy GPU nào trống. Sẽ chạy theo cấu hình mặc định (có thể là CPU hoặc GPU đang bận).")
            
    except Exception as e:
        log.warning(f"Lỗi khi kiểm tra GPU tự động (có thể hệ thống không có lệnh nvidia-smi): {e}")


# ──────────────────────────────────────────────────────────────
# Model Unwrapping
# ──────────────────────────────────────────────────────────────

def get_model(model):
    """
    Lấy model gốc nếu model đang được bọc bởi DataParallel hoặc DDP.
    Giúp truy xuất các thuộc tính custom và lưu checkpoint nhất quán.
    """
    return model.module if hasattr(model, 'module') else model

# ──────────────────────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────────────────────

def set_seed(seed=42):
    """
    Set random seed for reproducibility across all libraries.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    log.info(f"Set random seed to {seed}")
