import json
import argparse
import ast
import pandas as pd
import os
import random

# Dataset Paths Configuration
DATA_PATHS = {
    '2WikimultihopQA': {
        'train': '/mnt/shared-storage-user/liyafu/runquan/2wikimultihop/data/train.jsonl',
        'test': '/mnt/shared-storage-user/liyafu/runquan/2wikimultihop/data/dev.jsonl'
    },
    'HotpotQA': {
        'train': '/mnt/shared-storage-user/liyafu/runquan/hotpot/hotpot_train_v1.1.jsonl',
        'test': '/mnt/shared-storage-user/liyafu/runquan/hotpot/hotpot_dev_distractor_v1.jsonl'
    },
    'MuSiQue': {
        'train': '/mnt/shared-storage-user/liyafu/runquan/musique/data/musique_full_v1.0_train.jsonl',
        'test': '/mnt/shared-storage-user/liyafu/runquan/musique/data/musique_full_v1.0_dev.jsonl'
    },
    'Halueval': {
        'train': '/mnt/shared-storage-user/liyafu/runquan/HaluEval/data/qa_data.json',
        'test': '/mnt/shared-storage-user/liyafu/runquan/HaluEval/data/qa_data.json'
    }
}

def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Unified Data Preprocessing")
    parser.add_argument('--dataset', type=str, required=True, help="The dataset to process")
    parser.add_argument('--type', type=str, required=True, help="Data split to process (train or test)")
    parser.add_argument('--size', type=int, default=None, 
                        help="Number of samples to randomly select. Defaults to all samples.")
    parser.add_argument('--seed', type=int, default=42, 
                        help="Random seed for sampling. Defaults to 42.")
    return parser.parse_args()

def normalize_documents(record, dataset_name):
    """
    Extracts and normalizes documents/context from the record into a standard list format:
    [[title, content], [title, content], ...]
    where content is a string.
    """
    documents = []
    
    if dataset_name in ['2WikimultihopQA', 'HotpotQA']:
        # Format: "context": [[title, [sent1, sent2]], ...]
        raw_docs = record.get('context')
        if raw_docs is None:
            raw_docs = record.get('documents')
            
        if isinstance(raw_docs, str):
            try:
                raw_docs = ast.literal_eval(raw_docs)
            except:
                raw_docs = []
        
        if raw_docs:
            for item in raw_docs:
                if isinstance(item, list) and len(item) >= 2:
                    title = item[0]
                    content_list = item[1]
                    # content_list is usually a list of sentences
                    if isinstance(content_list, list):
                        content_str = " ".join(str(s) for s in content_list)
                    else:
                        content_str = str(content_list)
                    documents.append([title, content_str])

    elif dataset_name == 'MuSiQue':
        # Format: "paragraphs": [{"title": "...", "paragraph_text": "..."}, ...]
        paragraphs = record.get('paragraphs')
        if paragraphs:
            for p in paragraphs:
                title = p.get('title', '')
                text = p.get('paragraph_text', '')
                documents.append([title, text])
    
    elif dataset_name == 'Halueval':
        # Format: "knowledge": "text..."
        knowledge = record.get('knowledge', '')
        if knowledge:
            documents.append(['Knowledge', knowledge])
                
    return documents

def build_prompt(question_text, normalized_docs, record=None, dataset_name=None):
    """
    Constructs the formatted prompt string from the question and normalized documents.
    """
    # Specific handling for HaluEval
    if dataset_name == 'Halueval' and record:
        right_answer = record.get('right_answer', '')
        hallucinated_answer = record.get('hallucinated_answer', '')
        candidate_answers = [right_answer, hallucinated_answer]
        random.shuffle(candidate_answers)
        
        # In HaluEval, normalized_docs contains [['Knowledge', knowledge_text]]
        knowledge_text = ""
        if normalized_docs:
            knowledge_text = normalized_docs[0][1] # Get content from first doc
            
        references_context = f"Knowledge: {knowledge_text}" if knowledge_text else "No knowledge provided."
        
        final_prompt = f"""These are the documents you can reference:
{references_context}

Now answer the question:
{question_text}

Choose one answer from the following options:
- {candidate_answers[0]}
- {candidate_answers[1]}"""
        return final_prompt

    # Default handling for other datasets
    formatted_docs = []
    for title, content in normalized_docs:
        formatted_docs.append(f"Document '{title}': {content}")

    # Assemble the final prompt
    refs_block = "\n".join(formatted_docs) if formatted_docs else "No references provided."
    final_prompt = f"These are the documents you can reference:\n{refs_block}\n\nNow answer the question:\n{question_text}"
    
    return final_prompt

def transform_data(dataset_name, split_type, size=None, seed=42):
    """
    Reads data, transforms it, and saves to Parquet.
    """
    # 1. Determine Input Path
    source_path = DATA_PATHS.get(dataset_name, {}).get(split_type)
    if not source_path:
        print(f"Error: No path configured for dataset '{dataset_name}' split '{split_type}'")
        return

    print(f"Processing {dataset_name} ({split_type}) from: {source_path}")
    
    all_records = []

    try:
        # Load all data first to support random sampling
        if dataset_name == 'Halueval':
            full_data = []
            with open(source_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        full_data.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
                
            # Apply dataset-specific splitting first
            if split_type == 'train':
                all_records = full_data[:5000]
            else:
                all_records = full_data[5000:7000]
        else:
            with open(source_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        all_records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except FileNotFoundError:
        print(f"Error: Input file not found at {source_path}")
        return

    # 2. Random Sampling if size is specified
    total_available = len(all_records)
    print(f"Total available records: {total_available}")
    
    if size is not None and size < total_available:
        print(f"Sampling {size} records with seed {seed}...")
        random.seed(seed)
        records_to_process = random.sample(all_records, size)
    else:
        if size is not None:
            print(f"Requested size {size} >= total records {total_available}. Using all records.")
        records_to_process = all_records

    # 3. Process records
    extracted_data = []
    
    for i, record in enumerate(records_to_process):
        # Get Question
        query = record.get('question', '')

        # Get Answer
        if dataset_name == 'Halueval':
            ground_truth = record.get('right_answer', '')
        else:
            ground_truth = record.get('answer', '')

        # Normalize Documents
        normalized_docs = normalize_documents(record, dataset_name)
        documents_str = str(normalized_docs)

        # Generate Prompt
        prompt = build_prompt(query, normalized_docs, record, dataset_name)

        # Store the 4 required fields
        extracted_data.append({
            "prompt": prompt,
            "query": query,
            "ground_truth": ground_truth,
        })
                
    # Save to Parquet
    if extracted_data:
        df = pd.DataFrame(extracted_data)
        
        # Determine output path
        file_name = 'train.parquet' if split_type == 'train' else 'test.parquet'
        output_dir = os.path.join('data', dataset_name)
        output_path = os.path.join(output_dir, file_name)
        
        os.makedirs(output_dir, exist_ok=True)
            
        df.to_parquet(output_path, index=False)
        print(f"Saved {len(df)} records to {output_path}")
    else:
        print("No data extracted.")

def main():
    args = get_args()
    transform_data(args.dataset, args.type, args.size, args.seed)

if __name__ == "__main__":
    main()
