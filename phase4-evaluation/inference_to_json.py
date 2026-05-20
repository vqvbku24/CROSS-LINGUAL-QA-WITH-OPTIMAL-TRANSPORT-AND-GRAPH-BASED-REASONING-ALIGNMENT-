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
    Với câu is_impossible=True → trả về "" (đúng chuẩn SQuAD2).
    """
    if item.get("is_impossible", False):
        return ""

    val = item.get("answer") or item.get("answers")
    if val is None:
        return ""

    if isinstance(val, str):
        return val.strip()

    if isinstance(val, dict):
        texts = val.get("text", [])
        return texts[0].strip() if texts else ""

    if isinstance(val, list):
        return val[0].strip() if val else ""

    return ""


def find_best_span(start_logits, end_logits, K, max_span_len, keep_idx, question_end):
    """
    Tìm span (s, e) tối ưu trong CONTEXT (không phải question).
    Ràng buộc độ dài và thứ tự phải dựa trên TOKEN INDEX (không phải GRAPH NODE INDEX).
    """
    best_score = float('-inf')
    best_s, best_e = 0, 0
    found_context_span = False

    for s in range(K):
        for e in range(K):
            tok_s = keep_idx[s].item()
            tok_e = keep_idx[e].item()

            # Bỏ qua nếu token nằm trong phần câu hỏi
            if tok_s <= question_end + 1 or tok_e <= question_end + 1:
                continue

            # Ràng buộc trên TOKEN SPACE: start phải đứng trước end và không quá dài
            # Đã sửa `<` thành `<=` để cho phép đáp án 1 token (VD: "Paris")
            if tok_s <= tok_e and (tok_e - tok_s) <= max_span_len:
                score = start_logits[s].item() + end_logits[e].item()
                if score > best_score:
                    best_score = score
                    best_s, best_e = s, e
                    found_context_span = True

    return best_s, best_e, best_score, found_context_span


def is_unanswerable(start_logits, end_logits, best_span_score, na_threshold=0.0):
    """
    Fallback logic quyết định câu hỏi unanswerable dựa trên CLS score (cho checkpoint cũ).
    """
    cls_score = start_logits[0].item() + end_logits[0].item()
    return cls_score > best_span_score + na_threshold


def decode_span(input_ids, keep_idx, best_s, best_e, tokenizer):
    """
    Ánh xạ span từ graph-space về token-space rồi decode ra text sạch.
    Giữ nguyên thứ tự node gốc (KHÔNG sort) trùng khớp với GAT Encoder lúc train.
    """
    start_tok = keep_idx[best_s].item()
    end_tok   = keep_idx[best_e].item()

    pred_ids = input_ids[0, start_tok : end_tok + 1]
    return tokenizer.decode(pred_ids, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser(description="Inference mô hình Cross-Lingual QA ra file JSON")
    parser.add_argument("--checkpoint",  type=str, required=True,
                        help="Đường dẫn đến file .pt (VD: checkpoints/epoch_014.pt)")
    parser.add_argument("--input_file",  type=str, required=True,
                        help="File JSON chứa tập test (SQuAD format)")
    parser.add_argument("--output_file", type=str, default="phase4-evaluation/predictions.json",
                        help="File JSON kết quả đầu ra")
    parser.add_argument("--model_name",  type=str, default="xlm-roberta-base",
                        help="Tên model base (mặc định xlm-roberta-base)")
    parser.add_argument("--max_span_len", type=int, default=30,
                        help="Độ dài span tối đa (tính theo token gốc)")
    parser.add_argument("--na_threshold", type=float, default=2.0,
                        help="Fallback threshold cho các checkpoint không có has_answer_head")
    parser.add_argument("--debug", action="store_true", help="In chi tiết 5 sample đầu")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")

    # 🛠 KHÓA CỨNG: Luôn dùng xlm-roberta-base cho tokenizer để không bị lỗi config.json trên HuggingFace
    print("Đang load tokenizer (xlm-roberta-base)...")
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base", use_fast=True)

    print("Đang load dữ liệu...")
    try:
        dataset = load_squad_data(args.input_file)
    except Exception as e:
        print(f"Lỗi khi đọc file dữ liệu: {e}")
        return

    print(f"Đã load {len(dataset)} câu hỏi.")

    # Sanity check mẫu đầu tiên
    sample_gt = extract_ground_truth(dataset[0])
    print(f"🔍 Sanity check ground_truth[0]: '{sample_gt}'")

    print(f"Đang load trọng số mô hình từ {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint.get("config", {})

    K          = config.get("K",          128)
    gat_hidden = config.get("gat_hidden", 512)
    gat_out    = config.get("gat_out",    256)
    gat_layers = config.get("gat_layers", 2)

    print(f"   Config định hình: K={K}, gat_hidden={gat_hidden}, gat_out={gat_out}, gat_layers={gat_layers}")

    # 🛠 KHÓA CỨNG: Luôn khởi tạo mạng lưới bằng xlm-roberta-base, weights sẽ được nạp sau
    model = CrossLingualOTModel(
        model_name="xlm-roberta-base", K=K, gat_hidden=gat_hidden, gat_out=gat_out, gat_layers=gat_layers
    ).to(device)

    # Khởi tạo Criterion chứa lớp QAHead mới
    criterion = OTAlignmentLoss(
        qa_hidden_size=gat_out, K=K, q_hidden_size=model.backbone.hidden_size
    ).to(device)

    # 🛠 TIỀN XỬ LÝ WEIGHTS: Xóa tiền tố 'module.' nếu model được train bằng DataParallel
    model_state = checkpoint["model_state"]
    model_state_clean = {k.replace("module.", ""): v for k, v in model_state.items()}

    # Nạp trọng số cốt lõi và lấy thông báo
    load_result = model.load_state_dict(model_state_clean, strict=False)
    
    # 🚨 BỘ QUÉT LỖI TRỌNG SỐ (CỰC KỲ QUAN TRỌNG) 🚨
    print("\n[KIỂM TRA TRỌNG SỐ MODEL CHÍNH - BACKBONE & GAT]")
    missing = load_result.missing_keys
    print(f" - Số lượng key bị thiếu (Missing keys): {len(missing)}")
    if len(missing) > 0:
        print(" ⚠️ CẢNH BÁO: Model của bạn không có đủ trọng số của Backbone!")
        print(f" - VD một vài key thiếu: {missing[:5]}")
    else:
        print(" ✅ Tuyệt vời! Trọn bộ trọng số Backbone và GAT đã được nạp thành công.")
    
    # ── KHÓA CHẶT KIỂM TRA LỚP QA HEAD BẰNG STRICT=TRUE ──────────────────────────
    try:
        criterion_state = checkpoint["criterion_state"]
        criterion_state_clean = {k.replace("module.", ""): v for k, v in criterion_state.items()}
        criterion.load_state_dict(criterion_state_clean, strict=True)
        print(" ✅ Xác thực thành công: Đã nạp trọn vẹn lớp QAHead & HasAnswerHead!")
    except RuntimeError as e:
        print("\n❌ LỖI: Cấu trúc trọng số trong Checkpoint không đồng bộ với mã nguồn hiện tại!")
        print(f"Chi tiết kỹ thuật từ PyTorch:\n{e}")
        sys.exit(1)
    # ──────────────────────────────────────────────────────────────────────────

    model.eval()
    criterion.eval()

    print(f"\nBắt đầu suy luận (Inference) cho {len(dataset)} câu hỏi...")
    results = []

    for item in tqdm(dataset, desc="Đang dự đoán"):
        question     = item["question"]
        context      = item["context"]
        ground_truth = extract_ground_truth(item)

        input_ids, attention_mask, _, _, question_end = process_qa_sample(
            question=question, context=context, answer=None,
            tokenizer=tokenizer, max_length=512, doc_stride=128
        )

        input_ids      = input_ids.unsqueeze(0).to(device)
        attention_mask = attention_mask.unsqueeze(0).to(device)
        question_end   = question_end.item()
        seq_len        = input_ids.shape[1]

        with torch.no_grad():
            # Trích xuất đặc trưng từ backbone XLM-R
            hidden, attn = model.backbone(input_ids, attention_mask)

            # Phân tách đồ thị (VI Side không truyền nhãn đáp án -> [])
            q_idx = list(range(0, question_end + 1))
            sub_matrix, keep_idx = conditional_subsample(attn[0], q_idx, [], K=K)

            # Đẩy qua GAT Encoder
            feat = hidden[0, keep_idx, :]
            node_emb, _ = model.gat(feat, sub_matrix)

            # Chuẩn bị dữ liệu cho Cross-Attention trong QAHead
            q_emb = hidden[:, :question_end + 1, :]
            q_mask = torch.zeros(1, question_end + 1, dtype=torch.bool, device=device)

            # Thực thi QA Head dự đoán đầu ra
            qa_out = criterion.qa_head(node_emb.unsqueeze(0), q_emb, q_mask)
            start_logits      = qa_out[0].squeeze(0)
            end_logits        = qa_out[1].squeeze(0)
            has_answer_logit  = qa_out[2].squeeze(0) if len(qa_out) > 2 else None

            # Tìm kiếm Span tốt nhất trong không gian ngữ cảnh hợp lệ
            best_s, best_e, best_span_score, found_context_span = find_best_span(
                start_logits, end_logits, K, args.max_span_len, keep_idx, question_end
            )

            # Định đoạt dựa trên Dual-path Classifier Head
            if has_answer_logit is not None:
                is_ans = has_answer_logit.item() > 0
            else:
                is_ans = not is_unanswerable(start_logits, end_logits, best_span_score, na_threshold=args.na_threshold)

            if is_ans and found_context_span:
                predicted_answer = decode_span(input_ids, keep_idx, best_s, best_e, tokenizer)
            else:
                predicted_answer = ""

        results.append({
            "id":           item.get("id", str(len(results))),
            "question":     question,
            "answer":       predicted_answer,
            "ground_truth": ground_truth,
        })

        if args.debug and len(results) <= 5:
            cls_score = start_logits[0].item() + end_logits[0].item()
            start_tok = keep_idx[best_s].item() if best_s < len(keep_idx) else -1
            end_tok   = keep_idx[best_e].item() if best_e < len(keep_idx) else -1
            has_ans_prob = torch.sigmoid(has_answer_logit).item() if has_answer_logit is not None else float('nan')
            
            print(f"\n[DEBUG #{len(results)}]")
            print(f"  Q              : {question[:70]}")
            print(f"  GT             : '{ground_truth}'")
            print(f"  Pred           : '{predicted_answer}'")
            print(f"  has_answer_prob: {has_ans_prob:.3f} ({'answerable' if has_ans_prob > 0.5 else 'unanswerable'})")
            print(f"  cls_score      : {cls_score:.3f}  best_span_score: {best_span_score:.3f}")
            print(f"  best_s={best_s} (tok={start_tok})  best_e={best_e} (tok={end_tok})")
            print(f"  start stats    : min={start_logits.min():.3f} max={start_logits.max():.3f}  CLS={start_logits[0]:.3f}")
            print(f"  end   stats    : min={end_logits.min():.3f} max={end_logits.max():.3f}  CLS={end_logits[0]:.3f}")

    # Xuất kết quả cuối cùng
    out_dir = os.path.dirname(args.output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    empty_pred  = sum(1 for r in results if not r["answer"])
    empty_truth = sum(1 for r in results if not r["ground_truth"])
    print(f"\nThống kê nhanh:")
    print(f"  Predictions rỗng  : {empty_pred}/{len(results)} ({empty_pred/len(results)*100:.1f}%)")
    print(f"  Ground truth rỗng : {empty_truth}/{len(results)} ({empty_truth/len(results)*100:.1f}%)")
    print(f"\nHoàn thành! Đã lưu tại: {args.output_file}")


if __name__ == "__main__":
    main()