#!/usr/bin/env python3
"""
Generate SFT data for R-Tuning baseline.
Input: data/musique/train.parquet (or similar)
Output: SFT data (parquet/jsonl) with keys "prompt" and "response".

Logic:
1. Load dataset.
2. For answerable=True: response = "{answer}. I am sure."
3. For answerable=False: 
   - Inference with SYSTEM_PROMPT_TYPE=directly
   - response = "{model_output}. I am unsure."
"""

import pandas as pd
import json
import os
import sys
import numpy as np
import argparse
from tqdm import tqdm

# Add parent directory to path to import verl modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def series_to_item(ls):
    """
    Unwrap pandas Series or numpy arrays to get the actual data.
    """
    import pandas, numpy
    while isinstance(ls, (pandas.core.series.Series, numpy.ndarray)) and len(ls) == 1:
        ls = ls[0]
    return ls

def get_prompt_text(row):
    """
    Extract prompt text from a row, handling various formats.
    """
    prompt_raw = row.get('prompt', [])
    prompt = series_to_item(prompt_raw)
    
    prompt_text = ''
    
    if isinstance(prompt, dict):
        prompt_text = prompt.get('content', '')
    elif isinstance(prompt, list) and len(prompt) > 0:
        first_elem = prompt[0]
        if isinstance(first_elem, dict):
            prompt_text = first_elem.get('content', '')
        elif isinstance(first_elem, str):
            prompt_text = first_elem
    elif isinstance(prompt, str):
        try:
            prompt_parsed = json.loads(prompt)
            if isinstance(prompt_parsed, list) and len(prompt_parsed) > 0:
                prompt_text = prompt_parsed[0].get('content', '') if isinstance(prompt_parsed[0], dict) else ''
            elif isinstance(prompt_parsed, dict):
                prompt_text = prompt_parsed.get('content', '')
            else:
                prompt_text = prompt
        except:
            prompt_text = prompt
            
    return prompt_text

def parse_args():
    parser = argparse.ArgumentParser(description="Generate R-Tuning SFT data")
    parser.add_argument("--input-file", type=str, required=True, help="Input parquet file")
    parser.add_argument("--output-file", type=str, required=True, help="Output file")
    parser.add_argument("--model-path", type=str, required=True, help="Model path for inference")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--model-template", type=str, default="qwen")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Loading data from {args.input_file}")
    df = pd.read_parquet(args.input_file)
    print(f"Total samples: {len(df)}")
    
    # Identify answerable status
    answerable_mask = []
    for idx, row in df.iterrows():
        extra_info = row.get('extra_info', {})
        if isinstance(extra_info, str):
            try:
                extra_info = json.loads(extra_info)
            except:
                extra_info = {}
        
        # Handle case where extra_info might be wrapped
        extra_info = series_to_item(extra_info)
        if not isinstance(extra_info, dict):
            extra_info = {}
            
        answerable = extra_info.get('answerable', False)
        answerable_mask.append(answerable)
    
    df['is_answerable'] = answerable_mask
    
    df_true = df[df['is_answerable'] == True].copy()
    df_false = df[df['is_answerable'] == False].copy()
    
    print(f"Answerable samples: {len(df_true)}")
    print(f"Unanswerable samples: {len(df_false)}")
    
    results = []
    
    # Process Answerable Samples
    print("Processing answerable samples...")
    for idx, row in tqdm(df_true.iterrows(), total=len(df_true)):
        prompt_text = get_prompt_text(row)
        answer = series_to_item(row.get('answer', ''))
        
        # Ensure answer is string
        if not isinstance(answer, str):
            answer = str(answer)
            
        response = f"{answer}. I am sure."
        results.append({
            "prompt": prompt_text,
            "response": response
        })
        
    # Process Unanswerable Samples
    if not df_false.empty:
        print("Processing unanswerable samples with inference...")
        
        # Set environment variable for system prompt BEFORE import if needed, 
        # but wrap_prompt_with_system reads it at runtime.
        # However, to be safe and consistent with user request:
        os.environ['SYSTEM_PROMPT_TYPE'] = 'directly'
        from verl.utils.dataset.system_prompts import wrap_prompt_with_system
        
        # Extract prompts
        raw_prompts = []
        wrapped_prompts = []
        
        for idx, row in df_false.iterrows():
            p_text = get_prompt_text(row)
            raw_prompts.append(p_text)
            wrapped_prompts.append(wrap_prompt_with_system(p_text, model_template=args.model_template))
            
        # Initialize vLLM
        from vllm import LLM, SamplingParams
        
        print(f"Loading model from {args.model_path}")
        llm = LLM(
            model=args.model_path,
            trust_remote_code=True,
            dtype="bfloat16",
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len
        )
        
        sampling_params = SamplingParams(
            temperature=0,
            max_tokens=1024, # Reasonable limit for direct answers
            top_p=1.0,
            top_k=-1
        )
        
        print(f"Running inference on {len(wrapped_prompts)} samples...")
        outputs = llm.generate(wrapped_prompts, sampling_params)
        
        for i, output in enumerate(outputs):
            generated_text = output.outputs[0].text.strip()
            # Remove trailing period if present to avoid double period?? 
            # User example: "1991. I am sure."
            # If generated text is "1991.", output becomes "1991.. I am unsure."
            # I will just strictly follow format: "{zzz}. I am unsure."
            response = f"{generated_text}. I am unsure."
            
            results.append({
                "prompt": raw_prompts[i],
                "response": response
            })
            
    # Save results
    print(f"Saving {len(results)} results to {args.output_file}")
    output_df = pd.DataFrame(results)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    
    if args.output_file.endswith('.parquet'):
        output_df.to_parquet(args.output_file)
    else:
        output_df.to_json(args.output_file, orient='records', lines=True, force_ascii=False)
        
    print("Done!")

if __name__ == "__main__":
    main()

