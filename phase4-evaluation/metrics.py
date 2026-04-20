import json
import string
import collections
from underthesea import word_tokenize 

def normalize_vietnamese_text(text):
    if not text or not isinstance(text, str):
        return ""
    text = text.lower()
    exclude = set(string.punctuation)
    text = ''.join(ch for ch in text if ch not in exclude)
    text_tokenized = word_tokenize(text, format="text")
    return ' '.join(text_tokenized.split())

def exact_match_score(prediction, ground_truth):
    pred_norm = normalize_vietnamese_text(prediction)
    truth_norm = normalize_vietnamese_text(ground_truth)
    return int(pred_norm == truth_norm)

def f1_score(prediction, ground_truth):
    pred_tokens = normalize_vietnamese_text(prediction).split()
    truth_tokens = normalize_vietnamese_text(ground_truth).split()
    
    common = collections.Counter(pred_tokens) & collections.Counter(truth_tokens)
    num_same = sum(common.values())
    
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        val = float(pred_tokens == truth_tokens)
        return val, val, val
    if num_same == 0:
        return 0.0, 0.0, 0.0
        
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    
    return f1, precision, recall

def evaluate_json_file(file_path):
    print(f"Đang đọc dữ liệu từ: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file {file_path}")
        return

    # --- ĐOẠN XỬ LÝ CẤU TRÚC JSON ĐÃ ĐƯỢC CẬP NHẬT ---
    # Nếu dữ liệu là một Dictionary (như định dạng file của bạn: {"2": {...}, "3": {...}})
    if isinstance(raw_data, dict):
        # Chúng ta chỉ lấy phần values (các object chứa question, answer) bỏ qua key "2", "3"
        data = list(raw_data.values())
    elif isinstance(raw_data, list):
        data = raw_data
    else:
        data = [raw_data]

    total_em = 0
    total_f1 = 0
    total_samples = len(data)

    print(f"Bắt đầu chấm điểm cho {total_samples} câu hỏi với Underthesea Tokenizer...\n")

    for idx, item in enumerate(data):
        prediction = item.get('answer', '')
        ground_truth = item.get('ground_truth', '')
        
        em = exact_match_score(prediction, ground_truth)
        f1, precision, recall = f1_score(prediction, ground_truth)
        
        total_em += em
        total_f1 += f1
        
        # In chi tiết 2 câu đầu tiên để Sanity Check
        if idx < 2:
            print(f"--- TEST CÂU {idx + 1} ---")
            print(f"Tokenized Truth : {normalize_vietnamese_text(ground_truth)}")
            print(f"Tokenized Pred  : {normalize_vietnamese_text(prediction)}")
            print(f"-> Precision : {precision:.4f} | Recall: {recall:.4f}")
            print(f"-> F1-Score  : {f1:.4f} | EM: {em}")
            print("-" * 30 + "\n")

    if total_samples == 0:
        print("Không tìm thấy dữ liệu hợp lệ để đánh giá.")
        return

    avg_em = (total_em / total_samples) * 100
    avg_f1 = (total_f1 / total_samples) * 100
    
    print("=" * 40)
    print("KẾT QUẢ ĐÁNH GIÁ TỔNG QUAN (VIETNAMESE OPTIMIZED):")
    print(f"Tổng số câu hỏi : {total_samples}")
    print(f"Exact Match (EM): {avg_em:.2f}%")
    print(f"F1 Score        : {avg_f1:.2f}%")
    print("=" * 40)

if __name__ == "__main__":
    evaluate_json_file("Qwen_answer.json")