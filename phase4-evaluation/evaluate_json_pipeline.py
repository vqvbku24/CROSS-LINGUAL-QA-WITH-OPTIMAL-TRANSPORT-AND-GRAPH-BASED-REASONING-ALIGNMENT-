# evaluate_json_pipeline.py
import json
import argparse
from tqdm import tqdm
# Import các hàm chấm điểm từ file metrics.py nằm cùng thư mục
from metrics import exact_match_score, f1_score

def flatten_json_data(raw_data):
    """Hàm thông minh tự động ép mọi định dạng JSON về dạng List"""
    if isinstance(raw_data, dict):
        return list(raw_data.values())
    elif isinstance(raw_data, list):
        return raw_data
    return [raw_data]

def run_evaluation_pipeline(file_path):
    print(f"\n🚀 ĐANG KHỞI ĐỘNG PIPELINE ĐÁNH GIÁ...")
    print(f"📁 Nguồn dữ liệu: {file_path}")
    
    # Đọc file JSON
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
    except FileNotFoundError:
        print(f"❌ LỖI: Không tìm thấy file '{file_path}'")
        return
    except json.JSONDecodeError:
        print(f"❌ LỖI: File '{file_path}' bị sai định dạng JSON.")
        return

    # Chuẩn hóa dữ liệu
    data = flatten_json_data(raw_data)
    total_samples = len(data)

    if total_samples == 0:
        print("⚠️ CẢNH BÁO: File JSON trống, không có dữ liệu để đánh giá.")
        return

    print(f"✅ Đã tải thành công {total_samples} câu hỏi. Bắt đầu chấm điểm...\n")

    total_em = 0.0
    total_f1 = 0.0
    
    # Dùng tqdm để tạo thanh tiến trình cho ngầu
    for item in tqdm(data, desc="Đang xử lý (Underthesea)"):
        prediction = item.get('answer', '')
        ground_truth = item.get('ground_truth', '')
        
        # Tính điểm
        em = exact_match_score(prediction, ground_truth)
        f1, _, _ = f1_score(prediction, ground_truth)
        
        total_em += em
        total_f1 += f1

    # Tính trung bình và in báo cáo
    avg_em = (total_em / total_samples) * 100
    avg_f1 = (total_f1 / total_samples) * 100
    
    print("\n" + "=" * 50)
    print("📊 BÁO CÁO KẾT QUẢ ĐÁNH GIÁ (PIPELINE REPORT)")
    print("=" * 50)
    print(f"Tệp kiểm tra    : {file_path}")
    print(f"Tổng số câu hỏi : {total_samples}")
    print("-" * 50)
    print(f"🏆 Exact Match (EM) : {avg_em:05.2f}%")
    print(f"🎯 F1 Score         : {avg_f1:05.2f}%")
    print("=" * 50 + "\n")

if __name__ == "__main__":
    # Thiết lập argparse để nhận lệnh từ Terminal
    parser = argparse.ArgumentParser(description="Pipeline đánh giá F1/EM cho các file JSON QA")
    parser.add_argument(
        "--file", 
        type=str, 
        required=True, 
        help="Đường dẫn đến file JSON cần đánh giá"
    )
    
    args = parser.parse_args()
    run_evaluation_pipeline(args.file)