"""
metrics.py — Bộ chấm điểm chuẩn ViQuAD2 / SQuAD2
===================================================
Điểm khác biệt so với phiên bản cũ:
  1. normalize_vietnamese_text  : giữ nguyên, dùng underthesea
  2. f1_score / exact_match_score: API không đổi (tương thích ngược)
  3. best_f1_em (MỚI)           : tính max(F1/EM) trên nhiều gold answers,
                                  sau khi deduplicate → đúng chuẩn SQuAD2
  4. evaluate_viquad2_file (MỚI): đọc file SQuAD2-format gốc (.json với
                                  data/paragraphs/qas/is_impossible),
                                  xử lý cả câu answerable lẫn unanswerable
  5. evaluate_json_file         : giữ nguyên để tương thích với
                                  evaluate_json_pipeline.py (dùng file
                                  predictions.json dạng list)
"""

import json
import string
import collections
from underthesea import word_tokenize


# ─────────────────────────────────────────────────────────────
# 1. Chuẩn hóa văn bản
# ─────────────────────────────────────────────────────────────

def normalize_vietnamese_text(text: str) -> str:
    """Lower, bỏ dấu câu, word-tokenize bằng underthesea."""
    if not text or not isinstance(text, str):
        return ""
    text = text.lower()
    exclude = set(string.punctuation)
    text = ''.join(ch for ch in text if ch not in exclude)
    text_tokenized = word_tokenize(text, format="text")
    return ' '.join(text_tokenized.split())


# ─────────────────────────────────────────────────────────────
# 2. Điểm cơ bản (1 prediction vs 1 ground truth)
# ─────────────────────────────────────────────────────────────

def exact_match_score(prediction: str, ground_truth: str) -> int:
    return int(normalize_vietnamese_text(prediction) ==
               normalize_vietnamese_text(ground_truth))


def f1_score(prediction: str, ground_truth: str):
    """Trả về (f1, precision, recall)."""
    pred_tokens  = normalize_vietnamese_text(prediction).split()
    truth_tokens = normalize_vietnamese_text(ground_truth).split()

    common   = collections.Counter(pred_tokens) & collections.Counter(truth_tokens)
    num_same = sum(common.values())

    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        val = float(pred_tokens == truth_tokens)
        return val, val, val
    if num_same == 0:
        return 0.0, 0.0, 0.0

    precision = num_same / len(pred_tokens)
    recall    = num_same / len(truth_tokens)
    f1        = 2 * precision * recall / (precision + recall)
    return f1, precision, recall


# ─────────────────────────────────────────────────────────────
# 3. Điểm trên nhiều gold answers (chuẩn SQuAD2)
# ─────────────────────────────────────────────────────────────

def best_f1_em(prediction: str, gold_answers: list[str]):
    """
    Tính max(F1) và max(EM) trên tập gold_answers (đã deduplicate).

    Với câu unanswerable (gold_answers rỗng):
      - prediction == "" → EM=1, F1=1
      - prediction != "" → EM=0, F1=0

    Trả về (best_f1, best_em).
    """
    # Unanswerable
    if not gold_answers:
        if normalize_vietnamese_text(prediction) == "":
            return 1.0, 1
        return 0.0, 0

    # Deduplicate gold answers (giữ thứ tự)
    seen   = set()
    unique = []
    for ans in gold_answers:
        key = normalize_vietnamese_text(ans)
        if key not in seen:
            seen.add(key)
            unique.append(ans)

    best_f  = 0.0
    best_em = 0
    for gt in unique:
        f, _, _ = f1_score(prediction, gt)
        em      = exact_match_score(prediction, gt)
        if f  > best_f:  best_f  = f
        if em > best_em: best_em = em

    return best_f, best_em


# ─────────────────────────────────────────────────────────────
# 4. Đánh giá file SQuAD2 gốc (data/paragraphs/qas/...)
# ─────────────────────────────────────────────────────────────

def evaluate_viquad2_file(squad_file: str, predictions: dict):
    """
    Đánh giá chuẩn ViQuAD2/SQuAD2.

    Parameters
    ----------
    squad_file  : đường dẫn file .json gốc (có trường 'data')
    predictions : dict  {question_id: predicted_answer_str}
                  Với câu unanswerable, model nên predict "".

    Returns
    -------
    dict với avg_f1, avg_em (0-100), total, answerable, unanswerable
    """
    with open(squad_file, encoding='utf-8') as f:
        raw = json.load(f)

    articles = raw.get('data', [])

    total_f1 = 0.0
    total_em = 0
    n_total  = 0
    n_ans    = 0
    n_unans  = 0

    missing_ids = []

    for article in articles:
        for para in article['paragraphs']:
            for qa in para['qas']:
                qid          = qa['id']
                is_impossible = qa.get('is_impossible', False)
                gold_answers  = [a['text'] for a in qa.get('answers', [])]

                pred = predictions.get(qid, "")
                if qid not in predictions:
                    missing_ids.append(qid)

                f1, em = best_f1_em(pred, gold_answers)
                total_f1 += f1
                total_em += em
                n_total  += 1

                if is_impossible:
                    n_unans += 1
                else:
                    n_ans   += 1

    if missing_ids:
        print(f"⚠️  {len(missing_ids)} câu hỏi không có trong predictions dict "
              f"(sẽ tính là predict rỗng). VD: {missing_ids[:3]}")

    avg_f1 = (total_f1 / n_total) * 100 if n_total else 0.0
    avg_em = (total_em / n_total) * 100 if n_total else 0.0

    return {
        "avg_f1":       avg_f1,
        "avg_em":       avg_em,
        "total":        n_total,
        "answerable":   n_ans,
        "unanswerable": n_unans,
    }


# ─────────────────────────────────────────────────────────────
# 5. Đánh giá file predictions.json (list dạng cũ)
#    — giữ để tương thích với evaluate_json_pipeline.py
# ─────────────────────────────────────────────────────────────

def evaluate_json_file(file_path: str):
    """
    Đọc file predictions.json dạng list/dict (output của inference_to_json.py),
    mỗi item có 'answer' và 'ground_truth'.

    Dùng best_f1_em với 1 ground_truth (tương đương cách cũ nhưng đúng hơn
    ở chỗ xử lý unanswerable: ground_truth="" → pred="" → EM/F1=1).
    """
    print(f"Đang đọc dữ liệu từ: {file_path}")
    try:
        with open(file_path, encoding='utf-8') as f:
            raw_data = json.load(f)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file {file_path}")
        return
    except json.JSONDecodeError:
        print(f"Lỗi: File {file_path} sai định dạng JSON.")
        return

    if isinstance(raw_data, dict):
        data = list(raw_data.values())
    elif isinstance(raw_data, list):
        data = raw_data
    else:
        data = [raw_data]

    total_em = 0.0
    total_f1 = 0.0
    total_samples = len(data)

    print(f"Bắt đầu chấm điểm cho {total_samples} câu hỏi với Underthesea Tokenizer...\n")

    for idx, item in enumerate(data):
        prediction   = item.get('answer', '')
        ground_truth = item.get('ground_truth', '')

        # Xử lý ground_truth="" (unanswerable) đúng chuẩn
        gold_list = [ground_truth] if ground_truth else []
        f1, em    = best_f1_em(prediction, gold_list)

        total_em += em
        total_f1 += f1

        if idx < 2:
            f_val, p_val, r_val = f1_score(prediction, ground_truth) if ground_truth else (f1, f1, f1)
            print(f"--- TEST CÂU {idx + 1} ---")
            print(f"Tokenized Truth : {normalize_vietnamese_text(ground_truth)!r}")
            print(f"Tokenized Pred  : {normalize_vietnamese_text(prediction)!r}")
            print(f"-> Precision : {p_val:.4f} | Recall: {r_val:.4f}")
            print(f"-> F1-Score  : {f1:.4f}  | EM: {em}")
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