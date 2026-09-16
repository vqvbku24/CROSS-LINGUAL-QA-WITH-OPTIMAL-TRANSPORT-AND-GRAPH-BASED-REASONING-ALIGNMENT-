# Bản sao lưu gốc cho XLM-RoBERTa-base (XLM_base)

Thư mục này chứa toàn bộ các file mã nguồn nguyên bản (chưa qua chỉnh sửa) được trích xuất trực tiếp từ commit gốc `HEAD` của repo.

## Mục đích
- Giữ lại bản chuẩn tuyệt đối của hệ thống ban đầu chạy với `xlm-roberta-base`.
- Phục vụ so sánh, debug hoặc đối chiếu khi cần phục hồi logic cũ.

## Danh sách các file nguyên bản:
1. `phase2_model/model_core.py` (và `phase2_model/model_core_xlm_base.py`)
2. `train_stage1.py` (và `train_stage1_xlm_base.py`)
3. `train_stage2.py` (và `train_stage2_xlm_base.py`)
4. `data/xquad_loader.py` (và `data/xquad_loader_xlm_base.py`)
5. `hindi/data/xquad_loader_hi.py` (và `hindi/data/xquad_loader_hi_xlm_base.py`)
6. `hindi/train_stage2_hi.py` (và `hindi/train_stage2_hi_xlm_base.py`)
7. `arabic/data/xquad_loader_ar.py` (và `arabic/data/xquad_loader_ar_xlm_base.py`)
8. `arabic/train_stage2_ar.py` (và `arabic/train_stage2_ar_xlm_base.py`)

## Phiên bản hiện tại ở thư mục gốc:
- Các file ở thư mục chính (`model_core.py`, `train_stage1.py`, `train_stage2.py`, ...) là phiên bản đã được nâng cấp hỗ trợ cả `mmBERT` (ModernBERT) và `XLM-R` thông qua tham số `--model_name`.
