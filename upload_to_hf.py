import os
from dotenv import load_dotenv
from huggingface_hub import HfApi

def main():
    # Nạp biến môi trường từ .env
    load_dotenv()
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("❌ Không tìm thấy HF_TOKEN trong biến môi trường hoặc file .env")
        return
    
    api = HfApi(token=token)
    repo_id = "vinhvo1205/Sinkhorn_2_stages"
    
    # Các thư mục chứa checkpoint cần upload
    folders_to_upload = [
        "checkpoints"
    ]
    
    print(f"Bắt đầu đồng bộ các checkpoints lên repo: {repo_id}")
    
    for folder in folders_to_upload:
        if os.path.exists(folder) and os.path.isdir(folder):
            print(f"\n➤ Đang upload thư mục '{folder}'...")
            try:
                api.upload_folder(
                    folder_path=folder,
                    path_in_repo=folder, # Upload vào đúng thư mục cùng tên trên HF Hub
                    repo_id=repo_id,
                    repo_type="model"
                )
                print(f"✅ Đã upload thành công thư mục '{folder}'")
            except Exception as e:
                print(f"❌ Lỗi khi upload '{folder}': {e}")
        else:
            print(f"\n⚠️ Thư mục '{folder}' không tồn tại trong máy, bỏ qua.")
            
    print("\n🎉 Đã hoàn tất quá trình kiểm tra và upload!")

if __name__ == "__main__":
    main()
