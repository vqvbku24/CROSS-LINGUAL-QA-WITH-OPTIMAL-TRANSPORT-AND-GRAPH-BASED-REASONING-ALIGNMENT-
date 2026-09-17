import os
from dotenv import load_dotenv
from huggingface_hub import HfApi


def main():
    load_dotenv()

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("❌ Không tìm thấy HF_TOKEN")
        return

    api = HfApi(token=token)

    repo_id = "vinhvo1205/Sinkhorn_2_stages"

    local_file = "dataset/IndicSQuAD/train_hindi.json"
    path_in_repo = "dataset/IndicSQuAD/train_hindi.json"

    if not os.path.isfile(local_file):
        print(f"❌ Không tìm thấy file: {local_file}")
        return

    print(f"➤ Đang upload: {local_file}")
    print(f"➤ Repo: {repo_id}")
    print(f"➤ Path trên HF: {path_in_repo}")

    try:
        api.upload_file(
            path_or_fileobj=local_file,
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="model",
            commit_message="Upload IndicSQuAD Hindi training data",
        )

        print("✅ Upload thành công!")

    except Exception as e:
        print(f"❌ Upload thất bại:")
        print(f"   {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()