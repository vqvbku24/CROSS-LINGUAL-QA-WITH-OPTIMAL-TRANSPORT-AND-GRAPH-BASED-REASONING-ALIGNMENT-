import logging
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import json

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SquadParallelDataset(Dataset):
    """
    PyTorch Dataset cho dữ liệu song song Anh - Việt dựa trên SQuAD 2.0.
    """
    def __init__(self, parallel_data, tokenizer, max_length=384):
        self.parallel_data = parallel_data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.parallel_data)

    def __getitem__(self, idx):
        item = self.parallel_data[idx]
        en_item = item['en']
        vi_item = item['vi']

        # 1. Lấy ra question và context của cả bản EN và VI
        # Sử dụng .get() để phòng hờ trường hợp tên cột bị viết hoa chữ cái đầu
        en_question = en_item.get('question') or en_item.get('Question', '')
        en_context = en_item.get('context') or en_item.get('Context', '')
        
        vi_question = vi_item.get('question') or vi_item.get('Question', '')
        vi_context = vi_item.get('context') or vi_item.get('Context', '')

        # 2. Tokenize nhánh EN (Question + Context) qua process_qa_sample
        from phase1_dataloader.process_qa_sample import process_qa_sample
        en_answer = en_item.get('answer')
        is_answerable = en_answer is not None and len(en_answer.get("answer_start", [])) > 0
        
        en_input_ids, en_attention_mask, en_start_position, en_end_position, en_question_end = process_qa_sample(
            question=en_question,
            context=en_context,
            answer=en_answer if is_answerable else None,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            doc_stride=128
        )
        en_is_answerable = torch.tensor(1 if is_answerable else 0, dtype=torch.long)
        
        # 3. Tokenize nhánh VI tương tự như trên
        vi_encoding = self.tokenizer(
            text=vi_question,
            text_pair=vi_context,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        vi_input_ids = vi_encoding["input_ids"].squeeze(0)
        vi_attention_mask = vi_encoding["attention_mask"].squeeze(0)

        # 4. Tìm vị trí index của token [SEP] đầu tiên xuất hiện trong chuỗi của nhánh VI
        sep_id = self.tokenizer.sep_token_id
        
        def find_sep_idx(input_ids, sep_token_id):
            matches = (input_ids == sep_token_id).nonzero(as_tuple=True)[0]
            if len(matches) > 0:
                return matches[0].item()
            return 0

        vi_question_end = find_sep_idx(vi_input_ids, sep_id)

        # Định dạng Output của __getitem__
        return {
            "en_input_ids": en_input_ids,
            "en_attention_mask": en_attention_mask,
            "en_start_position": en_start_position,
            "en_end_position": en_end_position,
            "en_is_answerable": en_is_answerable,
            "en_question_end": en_question_end,
            "vi_input_ids": vi_input_ids,
            "vi_attention_mask": vi_attention_mask,
            "vi_question_end": torch.tensor(vi_question_end, dtype=torch.long)
        }

def create_squad_parallel_dataloaders(tokenizer, en_path="dataset/Squad2.0/train-v2.0.json", vi_path="dataset/AIForge_vietnamese-squad/train-00000-of-00001.parquet", batch_size=32, max_length=384, distributed=False):
    """
    Hàm tạo DataLoader cho dữ liệu song song từ local files.

    Args:
        distributed: If True, use DistributedSampler for DDP training.

    Returns:
        (train_loader, dataset, sampler_or_None)
    """
    # Bước 1: Tải dữ liệu tiếng Anh từ local json
    logger.info(f"Đang tải dữ liệu tiếng Anh từ {en_path}...")
    en_dict = {}
    with open(en_path, 'r', encoding='utf-8') as f:
        squad_data = json.load(f)['data']
        for article in squad_data:
            for paragraph in article['paragraphs']:
                context = paragraph['context']
                for qa in paragraph['qas']:
                    if qa.get("answers") and len(qa["answers"]) > 0:
                        first = qa["answers"][0]
                        answer_dict = {
                            "text": [first["text"]],
                            "answer_start": [int(first["answer_start"])],
                        }
                    else:
                        answer_dict = {"text": [], "answer_start": []}
                    en_dict[qa['id']] = {
                        'id': qa['id'],
                        'question': qa['question'],
                        'context': context,
                        'answer': answer_dict
                    }

    # Bước 2: Tải dữ liệu tiếng Việt từ local parquet file
    logger.info(f"Đang tải dữ liệu tiếng Việt từ {vi_path}...")
    vi_dataset = load_dataset("parquet", data_files=vi_path, split="train")

    # Bước 3: Dóng hàng (align) dựa trên cột id
    logger.info("Đang xử lý dóng hàng (alignment) dữ liệu dựa trên id...")
    parallel_data = []
    
    # Duyệt qua tập tiếng Việt để tra cứu id
    for vi_item in vi_dataset:
        vid = vi_item.get('id')
        if vid and vid in en_dict:
            parallel_data.append({
                'en': en_dict[vid],
                'vi': vi_item
            })

    # Log kết quả dóng hàng
    logger.info(f"Kết quả: Đã tìm thấy {len(parallel_data)} cặp dữ liệu song song khớp id.")
    
    if len(parallel_data) == 0:
        logger.warning("Không tìm thấy cặp dữ liệu nào khớp id! Hãy kiểm tra lại cột id của 2 dataset.")
        
    # Bước 3: Khởi tạo Dataset và DataLoader
    dataset = SquadParallelDataset(
        parallel_data=parallel_data, 
        tokenizer=tokenizer, 
        max_length=max_length
    )

    sampler = None
    if distributed:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(dataset, shuffle=True)
    
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),  # DistributedSampler handles shuffling
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
    )
    
    return train_loader, dataset, sampler
