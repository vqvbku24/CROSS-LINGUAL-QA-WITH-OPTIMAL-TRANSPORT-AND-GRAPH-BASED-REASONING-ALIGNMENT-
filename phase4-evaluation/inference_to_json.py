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
    # Ưu tiên check is_impossible trước
    if item.get("is_impossible", False):
        return ""

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


def find_best_span(start_logits, end_logits, K, max_span_len=30):
    """
    Tìm span (s, e) tối ưu với ràng buộc s <= e và e - s < max_span_len.

    ✅ FIX: Bắt đầu s từ 0 — khớp với training.
    Training dùng _remap_positions_to_graph_space (argmin nearest-neighbor)
    có thể map answer start/end vào bất kỳ node nào (kể cả node 0).
    Nếu skip node 0 ở inference → mất recall cho những sample đó.

    No-answer detection được xử lý riêng bởi is_unanswerable().

    Trả về (best_s, best_e, best_score) để caller có thể so sánh với
    no-answer score (cls_score) khi cần.
    """
    best_score = float('-inf')
    best_s, best_e = 0, 0

    for s in range(K):              # ← FIX: bắt đầu từ 0, khớp với training
        for e in range(s, min(s + max_span_len, K)):
            score = start_logits[s].item() + end_logits[e].item()
            if score > best_score:
                best_score = score
                best_s, best_e = s, e

    return best_s, best_e, best_score


def is_unanswerable(start_logits, end_logits, best_span_score, na_threshold=0.0):
    """
    Quyết định xem câu hỏi có unanswerable hay không.

    ✅ FIX Bug 3: Thay thế guard cứng "start_tok==0 → return ''"
    bằng so sánh score giữa no-answer (CLS) và best span.

    Logic:
      - cls_score = start_logits[0] + end_logits[0]  (SQuAD2 convention)
      - Nếu cls_score > best_span_score + na_threshold → unanswerable
      - na_threshold > 0 → thiên về answerable (tăng recall span)
      - na_threshold < 0 → thiên về unanswerable (tăng precision span)
    """
    cls_score = start_logits[0].item() + end_logits[0].item()
    return cls_score > best_span_score + na_threshold


def decode_span(input_ids, keep_idx, best_s, best_e, tokenizer):
    """
    Ánh xạ span từ graph-space về token-space rồi decode ra text.

    ✅ FIX: KHÔNG sort keep_idx.
    Training (model_core.py L73) dùng keep_idx nguyên bản (không sort)
    để tạo node embeddings: en_feat = en_hidden[i, en_keep, :]
    → QA Head học predict trên thứ tự node KHÔNG SORT.
    Nếu inference sort → mapping graph_node→token bị xáo trộn → decode sai.

    Quyết định unanswerable đã được xử lý trước bởi is_unanswerable().
    """
    # Dùng keep_idx nguyên bản — KHÔNG sort (khớp với training)
    start_tok = keep_idx[best_s].item()
    end_tok   = keep_idx[best_e].item()

    # Swap nếu cần (graph order có thể khác token order)
    if start_tok > end_tok:
        start_tok, end_tok = end_tok, start_tok

    pred_ids = input_ids[0, start_tok : end_tok + 1]
    return tokenizer.decode(pred_ids, skip_special_tokens=True).strip()


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
    parser.add_argument("--na_threshold", type=float, default=0.0,
                        help=(
                            "No-answer threshold: cls_score > best_span_score + threshold → predict ''. "
                            "Dương → thiên về answerable, âm → thiên về unanswerable. "
                            "Mặc định 0.0 (balanced). Tune sau khi có F1 baseline."
                        ))
    parser.add_argument("--debug", action="store_true",
                        help="In chi tiết 5 sample đầu để kiểm tra decode")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")

    print("Đang load tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    print("Đang load dữ liệu...")
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
        print("CẢNH BÁO: ground_truth mẫu đầu tiên rỗng! "
              "Nếu đây là câu answerable thì cần kiểm tra load_squad_data.")

    print(f"Đang load trọng số mô hình từ {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint.get("config", {})

    K          = config.get("K",          64)
    gat_hidden = config.get("gat_hidden", 512)
    gat_out    = config.get("gat_out",    256)
    gat_layers = config.get("gat_layers", 2)

    print(f"   Config: K={K}, gat_hidden={gat_hidden}, gat_out={gat_out}, gat_layers={gat_layers}")
    print(f"   NA threshold: {args.na_threshold:+.2f}")

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

    print(f"\nBắt đầu suy luận (Inference) cho {len(dataset)} câu hỏi...")
    results = []

    for item in tqdm(dataset, desc="Đang dự đoán"):
        question     = item["question"]
        context      = item["context"]
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

        seq_len = input_ids.shape[1]

        with torch.no_grad():
            # Backbone XLM-R
            hidden, attn = model.backbone(input_ids, attention_mask)

            # ✅ FIX: answer_indices = [] (trống) — khớp với training.
            # Training (model_core.py L70): VI side dùng answer_indices=[].
            # Inference cũng không có answer → phải truyền [] để graph
            # structure giống VI side lúc training.
            # Phiên bản cũ truyền toàn bộ ctx_idx vào answer_indices
            # khiến forced = q_idx + ctx_idx > K → graph bị cắt ngẫu nhiên
            # và cấu trúc hoàn toàn khác lúc train.
            q_idx = list(range(0, question_end + 1))
            sub_matrix, keep_idx = conditional_subsample(
                attn[0], q_idx, [], K=K  # ← answer_indices = []
            )

            # GAT Encoder
            feat = hidden[0, keep_idx, :]
            node_emb, _ = model.gat(feat, sub_matrix)

            # QA Head
            start_logits, end_logits = criterion.qa_head(node_emb.unsqueeze(0))
            start_logits = start_logits.squeeze(0)
            end_logits   = end_logits.squeeze(0)

            # ✅ FIX Bug 2: find_best_span bắt đầu từ node 1, trả về score
            best_s, best_e, best_span_score = find_best_span(
                start_logits, end_logits, K, args.max_span_len
            )

            # ✅ FIX Bug 3: Dùng no-answer threshold thay guard cứng
            if is_unanswerable(start_logits, end_logits, best_span_score,
                               na_threshold=args.na_threshold):
                predicted_answer = ""
            else:
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
            cls_score = start_logits[0].item() + end_logits[0].item()
            start_tok = keep_idx[best_s].item() if best_s < len(keep_idx) else -1
            end_tok   = keep_idx[best_e].item() if best_e < len(keep_idx) else -1
            print(f"\n[DEBUG #{len(results)}]")
            print(f"  Q         : {question[:70]}")
            print(f"  GT        : '{ground_truth}'")
            print(f"  Pred      : '{predicted_answer}'")
            print(f"  cls_score : {cls_score:.3f}  best_span_score: {best_span_score:.3f}")
            print(f"  best_s={best_s} (tok={start_tok})  best_e={best_e} (tok={end_tok})")
            print(f"  keep_idx [0:10]: {keep_idx[:10].tolist()}")
            print(f"  q_idx range: 0~{question_end}  seq_len: {seq_len}")

    # Xuất ra file JSON
    out_dir = os.path.dirname(args.output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    # --- Thống kê nhanh sau inference ---
    empty_pred  = sum(1 for r in results if not r["answer"])
    empty_truth = sum(1 for r in results if not r["ground_truth"])
    print(f"\nThống kê nhanh:")
    print(f"   Predictions rỗng  : {empty_pred}/{len(results)} ({empty_pred/len(results)*100:.1f}%)")
    print(f"   Ground truth rỗng : {empty_truth}/{len(results)} ({empty_truth/len(results)*100:.1f}%)")
    print(f"   (Ground truth rỗng lý tưởng = ~30.6% nếu dùng ViQuAD2 dev set)")

    print(f"\nHoàn thành! Đã lưu tại: {args.output_file}")
    print(f"Chạy evaluation (squad2 mode):")
    print(f"   python phase4-evaluation/evaluate_json_pipeline.py \\")
    print(f"       --mode squad2 \\")
    print(f"       --squad_file <dev_set.json> \\")
    print(f"       --pred_file  {args.output_file}")
    print(f"\nNếu F1 vẫn thấp, thử tune --na_threshold:")
    print(f"   +2.0  → model thiên về answerable hơn (tăng recall span)")
    print(f"   -2.0  → model thiên về unanswerable hơn (giảm false positives)")


if __name__ == "__main__":
    main()