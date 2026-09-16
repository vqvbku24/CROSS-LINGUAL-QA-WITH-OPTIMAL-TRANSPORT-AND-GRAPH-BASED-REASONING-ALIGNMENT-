import json
import pandas as pd
from pathlib import Path
import ast

def get_squad_contexts(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    contexts = set()
    if 'data' in data:
        if isinstance(data['data'], list):
            for article in data['data']:
                if isinstance(article, dict) and 'paragraphs' in article:
                    for paragraph in article['paragraphs']:
                        contexts.add(paragraph['context'].strip())
        elif isinstance(data['data'], dict):
            # Handle pandas orient='columns' format where 'data' is a dict of stringified dicts
            for k, article in data['data'].items():
                if isinstance(article, str):
                    try:
                        article_dict = ast.literal_eval(article)
                        if isinstance(article_dict, dict) and 'paragraphs' in article_dict:
                            for paragraph in article_dict['paragraphs']:
                                contexts.add(paragraph['context'].strip())
                    except Exception:
                        pass
                elif isinstance(article, dict):
                    if 'paragraphs' in article:
                        for paragraph in article['paragraphs']:
                            contexts.add(paragraph['context'].strip())
    return contexts

def get_jsonl_contexts(file_path):
    contexts = set()
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if 'context' in data:
                contexts.add(data['context'].strip())
    return contexts

def load_train_contexts(file_path):
    path = str(file_path)
    contexts = set()
    if not Path(path).exists():
        print(f"Warning: {path} not found.")
        return contexts
        
    if path.endswith('.parquet'):
        df = pd.read_parquet(path)
        if 'context' in df.columns:
            contexts.update(df['context'].str.strip().tolist())
    elif path.endswith('.json'):
        try:
            contexts = get_squad_contexts(path)
            if len(contexts) == 0: # Maybe huggingface dataset export as single JSON array
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            if 'context' in item:
                                contexts.add(item['context'].strip())
                    elif isinstance(data, dict) and 'context' in data:
                        # Probably pandas orient='columns' format
                        ctx_col = data['context']
                        if isinstance(ctx_col, dict):
                            for k, v in ctx_col.items():
                                if v is not None:
                                    contexts.add(str(v).strip())
        except Exception:
            contexts = get_jsonl_contexts(path)
    return contexts

def main():
    eval_files_vi = [
        'dataset/xquad.vi.json',
        'dataset/MLQA/test-context-vi-question-vi.json',
        'dataset/MLQA/test-context-en-question-vi.json',
        'dataset/MLQA/test-context-vi-question-en.json'
    ]
    train_file_vi = 'dataset/AIForge_vietnamese-squad/train-00000-of-00001.parquet'
    
    eval_files_ar = [
        'dataset/xquad.ar.json',
        'dataset/MLQA/test-context-ar-question-ar.json',
        'dataset/MLQA/test-context-en-question-ar.json',
        'dataset/MLQA/test-context-ar-question-en.json'
    ]
    train_file_ar = 'dataset/ZIZOUArabic_Squad/train.json'
    
    eval_files_hi = [
        'dataset/xquad.hi.json',
        'dataset/MLQA/test-context-hi-question-hi.json',
        'dataset/MLQA/test-context-en-question-hi.json',
        'dataset/MLQA/test-context-hi-question-en.json'
    ]
    train_file_hi = 'dataset/IndicSQuAD/train_hindi.json'
    
    def check_overlap(lang_name, eval_files, train_file):
        print(f"--- Checking {lang_name} ---")
        eval_contexts = set()
        for f in eval_files:
            if Path(f).exists():
                try:
                    eval_contexts.update(get_squad_contexts(f))
                except Exception as e:
                    print(f"Could not load {f}: {e}")
        
        train_contexts = load_train_contexts(train_file)
        
        if not eval_contexts or not train_contexts:
            print("Missing eval or train contexts. Skipping.")
            print()
            return
            
        overlap = eval_contexts.intersection(train_contexts)
        print(f"Eval unique contexts : {len(eval_contexts)}")
        print(f"Train unique contexts: {len(train_contexts)}")
        print(f"Overlap count        : {len(overlap)}")
        
        if len(overlap) > 0:
            print("Found overlaps!")
            # Print a few examples
            for i, o in enumerate(list(overlap)[:3]):
                print(f"  Example {i+1}: {o[:100]}...")
        else:
            print("=> NO OVERLAP FOUND!")
        print()

    check_overlap("Vietnamese (VI)", eval_files_vi, train_file_vi)
    check_overlap("Arabic (AR)", eval_files_ar, train_file_ar)
    check_overlap("Hindi (HI)", eval_files_hi, train_file_hi)

if __name__ == '__main__':
    main()
