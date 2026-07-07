import os
import subprocess
import logging
import torch

log = logging.getLogger(__name__)

def auto_select_free_gpus(memory_threshold_mb=1000):
    """
    Tự động tìm và gán CUDA_VISIBLE_DEVICES cho các GPU đang trống.
    Một GPU được coi là 'trống' nếu lượng VRAM đang sử dụng < memory_threshold_mb.
    Hàm này phải được gọi trước khi khởi tạo device trong PyTorch.
    """
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

def get_model(model):
    """
    Lấy model gốc nếu model đang được bọc bởi DataParallel.
    Giúp truy xuất các thuộc tính custom và lưu checkpoint nhất quán.
    """
    return model.module if hasattr(model, 'module') else model
