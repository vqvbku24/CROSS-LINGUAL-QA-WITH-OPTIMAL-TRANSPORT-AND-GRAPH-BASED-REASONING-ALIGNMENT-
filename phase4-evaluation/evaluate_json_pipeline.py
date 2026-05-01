"""
evaluate_json_pipeline.py — Pipeline đánh giá chuẩn ViQuAD2
=============================================================
Hỗ trợ 2 chế độ:

  Chế độ A (--mode squad2):  Đọc file SQuAD2-format gốc + file predictions
    python evaluate_json_pipeline.py \\
        --mode squad2 \\
        --squad_file  data/dev__public_test_.json \\
        --pred_file   phase4-evaluation/predictions.json

  Chế độ B (--mode legacy):  Đọc file predictions.json dạng cũ (list có
                               'answer' và 'ground_truth') — tương thích
                               với inference_to_json.py
    python evaluate_json_pipeline.py \\
        --mode legacy \\
        --file phase4-evaluation/predictions.json

Cải tiến so với phiên bản cũ:
  ✅ Xử lý is_impossible (unanswerable ~30% ViQuAD2)
  ✅ max(F1/EM) trên nhiều gold answers (dedup trước)
  ✅ Phân tích riêng answerable / unanswerable
  ✅ Cả 2 chế độ đều dùng best_f1_em từ metrics.py
"""

import json
import argparse
from tqdm import tqdm

from metrics import (
    best_f1_em,
    evaluate_viquad2_file,
    evaluate_json_file,
    normalize_vietnamese_text,
    f1_score,
    exact_match_score,
)


# ─────────────────────────────────────────────────────────────
# Helper: flatten predictions.json → {id: answer}
# ─────────────────────────────────────────────────────────────

def load_predictions_dict(pred_file: str) -> dict:
    """
    Đọc file predictions.json (output của inference_to_json.py).
    Trả về dict {id: answer_str}.
    """
    with open(pred_file, encoding='utf-8') as f:
        raw = json.load(f)

    if isinstance(raw, list):
        data = raw
    elif isinstance(raw, dict):
        # Có thể là dict-of-dicts hoặc {id: answer}
        first_val = next(iter(raw.values()))
        if isinstance(first_val, str):
            # Đã là {id: answer} rồi
            return raw
        data = list(raw.values())
    else:
        raise ValueError("Định dạng predictions file không hỗ trợ.")

    return {item["id"]: item.get("answer", "") for item in data}


# ─────────────────────────────────────────────────────────────
# Chế độ A: Squad2 (đánh giá chuẩn ViQuAD2)
# ─────────────────────────────────────────────────────────────

def run_squad2_pipeline(squad_file: str, pred_file: str):
    print(f"\n🚀 ĐANG KHỞI ĐỘNG PIPELINE ĐÁNH GIÁ (CHẾ ĐỘ SQUAD2)...")
    print(f"📁 Dev set  : {squad_file}")
    print(f"📁 Pred file: {pred_file}\n")

    # Load predictions
    try:
        predictions = load_predictions_dict(pred_file)
    except FileNotFoundError:
        print(f"❌ LỖI: Không tìm thấy file '{pred_file}'")
        return
    except Exception as e:
        print(f"❌ LỖI khi đọc predictions: {e}")
        return

    print(f"✅ Đã tải {len(predictions)} predictions.\n")

    # Chạy đánh giá chuẩn ViQuAD2
    try:
        result = evaluate_viquad2_file(squad_file, predictions)
    except FileNotFoundError:
        print(f"❌ LỖI: Không tìm thấy file '{squad_file}'")
        return

    # In báo cáo
    print("\n" + "=" * 55)
    print("📊 BÁO CÁO KẾT QUẢ ĐÁNH GIÁ — VIQUAD2 CHUẨN")
    print("=" * 55)
    print(f"Dev set        : {squad_file}")
    print(f"Predictions    : {pred_file}")
    print(f"Tổng câu hỏi  : {result['total']}")
    print(f"  Answerable   : {result['answerable']}")
    print(f"  Unanswerable : {result['unanswerable']}")
    print("-" * 55)
    print(f"🏆 Exact Match (EM) : {result['avg_em']:06.2f}%")
    print(f"🎯 F1 Score         : {result['avg_f1']:06.2f}%")
    print("=" * 55 + "\n")

    print("💡 Ghi chú:")
    print("   • F1/EM được tính theo max trên các gold answers đã dedup.")
    print("   • Unanswerable: predict '' → EM=1 F1=1, ngược lại → 0.")


# ─────────────────────────────────────────────────────────────
# Chế độ B: Legacy (file predictions.json dạng cũ)
# ─────────────────────────────────────────────────────────────

def flatten_json_data(raw_data):
    if isinstance(raw_data, dict):
        return list(raw_data.values())
    elif isinstance(raw_data, list):
        return raw_data
    return [raw_data]


def run_legacy_pipeline(file_path: str):
    print(f"\n🚀 ĐANG KHỞI ĐỘNG PIPELINE ĐÁNH GIÁ (CHẾ ĐỘ LEGACY)...")
    print(f"📁 Nguồn dữ liệu: {file_path}")

    try:
        with open(file_path, encoding='utf-8') as f:
            raw_data = json.load(f)
    except FileNotFoundError:
        print(f"❌ LỖI: Không tìm thấy file '{file_path}'")
        return
    except json.JSONDecodeError:
        print(f"❌ LỖI: File '{file_path}' bị sai định dạng JSON.")
        return

    data = flatten_json_data(raw_data)
    total_samples = len(data)

    if total_samples == 0:
        print("⚠️ CẢNH BÁO: File JSON trống, không có dữ liệu để đánh giá.")
        return

    print(f"✅ Đã tải thành công {total_samples} câu hỏi. Bắt đầu chấm điểm...\n")

    total_em = 0.0
    total_f1 = 0.0

    for item in tqdm(data, desc="Đang xử lý"):
        prediction   = item.get('answer', '')
        ground_truth = item.get('ground_truth', '')

        # Xử lý unanswerable đúng chuẩn
        gold_list = [ground_truth] if ground_truth else []
        f1, em    = best_f1_em(prediction, gold_list)

        total_em += em
        total_f1 += f1

    avg_em = (total_em / total_samples) * 100
    avg_f1 = (total_f1 / total_samples) * 100

    print("\n" + "=" * 55)
    print("📊 BÁO CÁO KẾT QUẢ ĐÁNH GIÁ (PIPELINE REPORT)")
    print("=" * 55)
    print(f"Tệp kiểm tra    : {file_path}")
    print(f"Tổng số câu hỏi : {total_samples}")
    print("-" * 55)
    print(f"🏆 Exact Match (EM) : {avg_em:06.2f}%")
    print(f"🎯 F1 Score         : {avg_f1:06.2f}%")
    print("=" * 55 + "\n")

    print("💡 Ghi chú:")
    print("   • Dùng best_f1_em: ground_truth='' được xử lý là unanswerable.")
    print("   • Để đánh giá multi-answer chính xác hơn, dùng --mode squad2.")


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pipeline đánh giá F1/EM chuẩn ViQuAD2",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["squad2", "legacy"],
        default="legacy",
        help=(
            "squad2 : đánh giá chuẩn ViQuAD2 (cần --squad_file + --pred_file)\n"
            "legacy : đọc file predictions.json dạng cũ (cần --file)"
        ),
    )

    # Squad2 args
    parser.add_argument("--squad_file", type=str,
                        help="[squad2] File SQuAD2-format gốc (dev set)")
    parser.add_argument("--pred_file",  type=str,
                        help="[squad2] File predictions.json từ inference_to_json.py")

    # Legacy args
    parser.add_argument("--file", type=str,
                        help="[legacy] File predictions.json cần đánh giá")

    args = parser.parse_args()

    if args.mode == "squad2":
        if not args.squad_file or not args.pred_file:
            parser.error("--mode squad2 yêu cầu cả --squad_file và --pred_file")
        run_squad2_pipeline(args.squad_file, args.pred_file)
    else:
        if not args.file:
            parser.error("--mode legacy yêu cầu --file")
        run_legacy_pipeline(args.file)