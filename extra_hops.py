import json
import pandas as pd
import os
import random
from preprocess import DATA_PATHS, normalize_documents, build_prompt

def extract_hops_data(seed=42):
    dataset_name = 'HotpotQA'
    split_type = 'train'
    
    # Get source path from preprocess config
    source_path = DATA_PATHS.get(dataset_name, {}).get(split_type)
    if not source_path:
        print(f"Error: No path configured for dataset '{dataset_name}' split '{split_type}'")
        return

    print(f"Reading from: {source_path}")
    
    # Store raw records by hop
    records_by_hop = {
        '2': [],
        '3': [],
        '4': []
    }
    
    try:
        with open(source_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    record = json.loads(line)
                    hop = None
                    if dataset_name == 'HotpotQA':
                        supporting_facts = record.get('supporting_facts', [])
                        hop = str(len(supporting_facts))
                    else:
                        rec_id = record.get('id', '')
                        if rec_id:
                            hop = rec_id[0]

                    if hop in records_by_hop:
                        records_by_hop[hop].append(record)
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        print(f"Error: Input file not found at {source_path}")
        return

    print("Records found:")
    for hop, recs in records_by_hop.items():
        print(f"  {hop}-hop: {len(recs)}")

    # Process and save 1000 samples for each hop
    output_dir = os.path.join('data', dataset_name, 'hops_split')
    os.makedirs(output_dir, exist_ok=True)
    
    random.seed(seed)
    target_count = 5000
    
    for hop, records in records_by_hop.items():
        if len(records) < target_count:
            print(f"Warning: Only found {len(records)} records for {hop}-hop (requested {target_count}). Using all available.")
            selected_records = records
        else:
            selected_records = random.sample(records, target_count)
            
        processed_data = []
        for record in selected_records:
            query = record.get('question', '')
            # Match preprocess.py logic for answer
            ground_truth = record.get('answer', '')
            
            normalized_docs = normalize_documents(record, dataset_name)
            prompt = build_prompt(query, normalized_docs, record, dataset_name)
            
            processed_data.append({
                "prompt": prompt,
                "query": query,
                "ground_truth": ground_truth,
                # "id": record.get('id'), # Optional: keep ID if needed
                # "hop": hop # Optional
            })
            
        output_path = os.path.join(output_dir, f'{hop}hop_1000.parquet')
        df = pd.DataFrame(processed_data)
        df.to_parquet(output_path, index=False)
        print(f"Saved {len(df)} records to {output_path}")

if __name__ == "__main__":
    extract_hops_data()

