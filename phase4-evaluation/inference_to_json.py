import os
import sys
import json
import torch
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer

# Thêm thư mục gốc vào đường dẫn hệ thống để import được các module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase2_model.model_core import CrossLingualOTModel
from phase3_loss.losses import OTAlignmentLoss
from phase1_dataloader.process_qa_sample import process_qa_sample, load_squad_data
from phase2_model.modules.subsampling import conditional_subsample


def extract_ground_truth(item):
    """
    Đọc ground truth linh hoạt, hỗ trợ 3 format:
      1. answer là string thẳng          -> "Thái Bình Dương"
      2. answer là dict SQuAD gốc        -> {"text": ["Thái Bình Dương"], "answer_start": [10]}
      3. answers là dict SQuAD gốc       -> {"text": [...], "answer_start": [...]}
    """
    val = item.get("answer") or item.get("answers")

    if val is None:
        return ""

    # Format 1: string thẳng (sau khi load_squad_data flatten)
    if isinstance(val, str):
        return val.strip()

    # Format 2 & 3: dict {"text": [...], "answer_start": [...]}
    if isinstance(val, dict):
        texts = val.get("text", [])
        return texts[0].strip() if texts else ""

    # Format hiếm: list string
    if isinstance(val, list):
        return val[0].strip() if val else ""

    return ""


def decode_span(input_ids, keep_idx, best_s, best_e, tokenizer):
    """
    Ánh xạ span từ graph-space về token-space rồi decode ra text.

    Vấn đề gốc rễ: conditional_subsample trả về keep_idx KHÔNG sorted theo
    thứ tự tăng dần của token position. Ví dụ keep_idx có thể là:
        [0, 5, 2, 8, 3, ...]  (xáo trộn)
    → best_s=1 (token 5), best_e=2 (token 2) → decode ra "mento" thay vì
      decode đúng span liên tục trong context.

    FIX: Sort keep_idx theo token position trước khi tìm span, đảm bảo
    graph node i luôn tương ứng với token có position nhỏ hơn graph node i+1.
    """
    # Sắp xếp keep_idx theo thứ tự token position tăng dần
    # keep_idx shape: (K,) hoặc Tensor 1-D
    sorted_keep, _ = keep_idx.sort()

    start_tok = sorted_keep[best_s].item()
    end_tok   = sorted_keep[best_e].item()

    # Đề phòng best_s > best_e sau khi sort (không nên xảy ra nhưng an toàn)
    if start_tok > end_tok:
        start_tok, end_tok = end_tok, start_tok

    # Token 0 là [CLS] → model predict unanswerable
    if start_tok == 0 and end_tok == 0:
        return ""

    pred_ids = input_ids[0, start_tok : end_tok + 1]
    return tokenizer.decode(pred_ids, skip_special_tokens=True).strip()


def find_best_span(start_logits, end_logits, K, max_span_len=30):
    """
    Tìm span (s, e) tối ưu với ràng buộc s <= e và e - s < max_span_len.
    Dùng vòng lặp O(K * max_span_len) thay vì O(K^2).
    """
    best_score = float('-inf')
    best_s, best_e = 0, 0

    for s in range(K):
        for e in range(s, min(s + max_span_len, K)):
            score = start_logits[s].item() + end_logits[e].item()
            if score > best_score:
                best_score = score
                best_s, best_e = s, e

    return best_s, best_e


def main():
    parser = argparse.ArgumentParser(description="Inference mô hình Cross-Lingual QA ra file JSON")
    parser.add_argument("--checkpoint",  type=str, required=True,
                        help="Đường dẫn đến file .pt (VD: checkpoints/epoch_010.pt)")
    parser.add_argument("--input_file",  type=str, required=True,
                        help="File JSON chứa tập test (SQuAD format)")
    parser.add_argument("--output_file", type=str,
                        default="phase4-evaluation/predictions.json",
                        help="File JSON kết quả đầu ra")
    parser.add_argument("--model_name",  type=str, default="xlm-roberta-base")
    parser.add_argument("--max_span_len", type=int, default=30,
                        help="Độ dài span tối đa (tính theo graph node, không phải token)")
    parser.add_argument("--debug", action="store_true",
                        help="In chi tiết 5 sample đầu để kiểm tra decode")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")

    print("Đang load tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    print(" Đang load dữ liệu...")
    try:
        dataset = load_squad_data(args.input_file)
    except Exception as e:
        print(f"Lỗi khi đọc {args.input_file}. Chi tiết: {e}")
        return

    print(f"Đã load {len(dataset)} câu hỏi.")

    # --- Sanity check ground truth ngay sau khi load ---
    sample_gt = extract_ground_truth(dataset[0])
    print(f"🔍 Sanity check ground_truth[0]: '{sample_gt}'")
    if not sample_gt:
        print("CẢNH BÁO: ground_truth mẫu đầu tiên rỗng! Kiểm tra lại format file input.")

    print(f" Đang load trọng số mô hình từ {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint.get("config", {})

    K          = config.get("K",          64)
    gat_hidden = config.get("gat_hidden", 512)
    gat_out    = config.get("gat_out",    256)
    gat_layers = config.get("gat_layers", 2)

    print(f"   Config: K={K}, gat_hidden={gat_hidden}, gat_out={gat_out}, gat_layers={gat_layers}")

    model = CrossLingualOTModel(
        model_name=args.model_name,
        K=K,
        gat_hidden=gat_hidden,
        gat_out=gat_out,
        gat_layers=gat_layers
    ).to(device)

    criterion = OTAlignmentLoss(
        qa_hidden_size=gat_out,
        K=K
    ).to(device)

    model.load_state_dict(checkpoint["model_state"], strict=False)
    criterion.load_state_dict(checkpoint["criterion_state"], strict=False)

    model.eval()
    criterion.eval()

    print(f"\n Bắt đầu suy luận (Inference) cho {len(dataset)} câu hỏi...")
    results = []

    for item in tqdm(dataset, desc="Đang dự đoán"):
        question     = item["question"]
        context      = item["context"]

        # ✅ FIX 1: Đọc ground truth đúng cách
        ground_truth = extract_ground_truth(item)

        # Tokenize (KHÔNG truyền answer vì đây là bước test)
        input_ids, attention_mask, _, _, question_end = process_qa_sample(
            question=question,
            context=context,
            answer=None,
            tokenizer=tokenizer,
            max_length=512,
            doc_stride=128
        )

        input_ids      = input_ids.unsqueeze(0).to(device)
        attention_mask = attention_mask.unsqueeze(0).to(device)
        question_end   = question_end.item()

        with torch.no_grad():
            # Backbone XLM-R
            hidden, attn = model.backbone(input_ids, attention_mask)

            # Subsampling
            q_idx = list(range(0, question_end + 1))
            sub_matrix, keep_idx = conditional_subsample(attn[0], q_idx, [], K=K)

            # GAT Encoder
            feat = hidden[0, keep_idx, :]
            node_emb, _ = model.gat(feat, sub_matrix)

            # QA Head
            start_logits, end_logits = criterion.qa_head(node_emb.unsqueeze(0))
            start_logits = start_logits.squeeze(0)
            end_logits   = end_logits.squeeze(0)

            # ✅ FIX 2: Tìm span tối ưu (tách thành hàm riêng, dễ debug)
            best_s, best_e = find_best_span(
                start_logits, end_logits, K, args.max_span_len
            )

            # ✅ FIX 2 (tiếp): Decode span về text
            predicted_answer = decode_span(
                input_ids, keep_idx, best_s, best_e, tokenizer
            )

        results.append({
            "id":           item.get("id", str(len(results))),
            "question":     question,
            "answer":       predicted_answer,
            "ground_truth": ground_truth,
        })

        # Debug 5 sample đầu
        if args.debug and len(results) <= 5:
            print(f"\n[DEBUG #{len(results)}]")
            print(f"  Q    : {question[:70]}")
            print(f"  Pred : '{predicted_answer}'")
            print(f"  keep_idx (unsorted, first 10): {keep_idx[:10].tolist()}")
            sorted_k, _ = keep_idx.sort()
            print(f"  keep_idx (sorted,   first 10): {sorted_k[:10].tolist()}")

    # Xuất ra file JSON
    out_dir = os.path.dirname(args.output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    # --- Thống kê nhanh sau inference ---
    empty_pred  = sum(1 for r in results if not r["answer"])
    empty_truth = sum(1 for r in results if not r["ground_truth"])
    print(f"\n Thống kê nhanh:")
    print(f"   Predictions rỗng  : {empty_pred}/{len(results)} ({empty_pred/len(results)*100:.1f}%)")
    print(f"   Ground truth rỗng : {empty_truth}/{len(results)} ({empty_truth/len(results)*100:.1f}%)")

    print(f"\n Hoàn thành! Đã lưu tại: {args.output_file}")
    print(f" Chạy evaluation:\n   python phase4-evaluation/evaluate_json_pipeline.py --file {args.output_file}")


if __name__ == "__main__":
    main()