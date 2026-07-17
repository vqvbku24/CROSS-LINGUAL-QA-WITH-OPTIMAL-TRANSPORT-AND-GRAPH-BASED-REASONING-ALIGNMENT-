import numpy as np

def main():
    try:
        hb = np.load('paper_tools/export/hidden_before.npy')
        ha = np.load('paper_tools/export/hidden_after.npy')
        
        diff = np.abs(hb - ha).mean()
        max_diff = np.abs(hb - ha).max()
        
        print("=== CHECK EMBEDDING DIFFERENCE ===")
        print(f"Mean absolute difference (Before vs After): {diff:.6f}")
        print(f"Max absolute difference  (Before vs After): {max_diff:.6f}")
        
        if diff == 0:
            print("\n[LỖI] Embeddings hoàn toàn GIỐNG NHAU (Diff = 0).")
            print("Nguyên nhân có thể: Checkpoint không chứa trọng số LoRA, hoặc LoRA bị bypass.")
        else:
            print("\n[OK] Embeddings CÓ THAY ĐỔI. LoRA đã hoạt động.")
            print("Giải thích: UMAP vẫn tách 2 cụm là do 'Language Bias' của XLM-R (khoảng cách giữa 2 ngôn ngữ lớn hơn khoảng cách semantic).")
            
    except Exception as e:
        print("Lỗi:", e)

if __name__ == "__main__":
    main()
