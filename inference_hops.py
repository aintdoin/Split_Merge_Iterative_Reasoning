#!/usr/bin/env python3
import argparse
import os
import sys
import json
import pandas as pd
import requests
import concurrent.futures
import re
import numpy as np
from tqdm import tqdm
from vllm import LLM, SamplingParams

# Add parent directory to path to import prompts.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from prompts import wrap_prompt_with_system
except ImportError:
    print("Error: prompts.py not found in the parent directory.")
    sys.exit(1)

class LLMJudge:
    def __init__(self, api_base, model_name, api_key, max_workers=8, timeout=60):
        self.api_base = api_base.rstrip('/')
        self.model_name = model_name
        self.api_key = api_key
        self.max_workers = int(max_workers)
        self.timeout = float(timeout)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)

    def _call_api(self, prompt):
        url = f"{self.api_base}/v1/chat/completions"
        headers = {'Content-Type': 'application/json'}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        
        payload = {
            'model': self.model_name,
            'messages': [
                {'role': 'system', 'content': 'You are a strict answer evaluator. Output only a single digit.'},
                {'role': 'user', 'content': prompt}
            ],
            'temperature': 0.0,
            'max_tokens': 5,
            'stream': False
        }
        
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
            if resp.status_code == 200:
                return resp.json()['choices'][0]['message']['content'].strip()
        except Exception as e:
            pass
        return "-1"

    def judge(self, question, prediction, ground_truth):
        # 1. Exact match (case-insensitive)
        if str(prediction).strip().lower() == str(ground_truth).strip().lower():
            return 1

        # 2. LLM Judge
        prompt = f"""Input: Assume you are a human expert in grading predictions given by a model. You are given a question and a model prediction. Judge if the prediction matches the ground truth answer by following these steps:
1: Take it as granted that the Ground Truth is always correct.
2: If the Prediction exactly matches the Ground Truth, “score" is 1.
3: If the Prediction does not exactly match the Ground Truth, go through the following steps and likely give a score as -1.
4: If the Ground Truth is a number, “score" is 1 if and only if the Prediction gives a number that almost exactly matches the ground truth.
5: If the Prediction is self-contradictory, “score" must be -1.
6: If the prediction is not answering the question, “score" must be -1.
7: If the prediction is a concise and correct summary of the ground truth, “score" is 1.
8: If ground truth contains a set of items, prediction must contain exactly same items for the score to be 1.
9: Otherwise, “score" is -1.

Output ONLY one digit: -1 or 1. No text, no explanation.

Question: {question}
Ground Truth: {ground_truth}
Prediction: {prediction}
Output: """
        
        try:
            future = self.executor.submit(self._call_api, prompt)
            result = future.result(timeout=self.timeout + 5)
            match = re.search(r'-?\d+', result)
            if match:
                score = int(match.group())
                return 1 if score == 1 else -1 # Map anything else to -1 (or keep -1)
            return -1
        except Exception:
            return -1

def extract_answer(text):
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip()

def count_steps(text):
    # Extract content within <think> tags if present, otherwise use full text
    match = re.search(r'<think>(.*?)</think>', text, re.DOTALL | re.IGNORECASE)
    content = match.group(1) if match else text
    
    # Regex based on verl/workers/fsdp_workers.py:1381
    # Matches numbered list items like "1. ", " 2. " at the start of lines
    enum_pattern = re.compile(r'(?m)^(\s*)(\d+)\.(\s*)')
    markers = list(enum_pattern.finditer(content))
    return len(markers)

def main():
    parser = argparse.ArgumentParser(description="Inference Script with Hop Analysis")
    parser.add_argument("--model", required=True, help="Path to the model")
    parser.add_argument("--datasets", required=True, help="Path to parquet dataset")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-output", type=int, default=1024)
    parser.add_argument("--prompt-template", default="cot", help="System prompt template name (e.g. cot, tot)")
    parser.add_argument("--model-template", default="qwen", help="Model template type (e.g. qwen, llama)")
    
    # LLM Judge Args
    parser.add_argument("--judge-api-base", required=True)
    parser.add_argument("--judge-model-name", required=True)
    parser.add_argument("--judge-api-key", default="")
    parser.add_argument("--judge-max-workers", default=8)
    parser.add_argument("--judge-timeout", default=60)
    
    args = parser.parse_args()

    # Set prompt template env var for prompts.py
    os.environ['SYSTEM_PROMPT_TYPE'] = args.prompt_template

    # Load Data
    print(f"Loading data from {args.datasets}...")
    df = pd.read_parquet(args.datasets)
    print(f"Loaded {len(df)} samples.")

    # Prepare Prompts
    prompts = []
    print("Preparing prompts...")
    for _, row in df.iterrows():
        # Combine system prompt + user prompt
        # row['prompt'] is the user content (question + refs)
        full_prompt = wrap_prompt_with_system(row['prompt'], args.model_template)
        prompts.append(full_prompt)

    # Init Model
    print(f"Initializing model: {args.model}")
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        tensor_parallel_size=1, # Default 1, can be exposed if needed
        max_model_len=16384 # Default large enough
    )
    
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_output,
        n=1
    )

    # Generate
    print("Generating responses...")
    outputs = llm.generate(prompts, sampling_params)

    # Init Judge
    judge = LLMJudge(
        args.judge_api_base,
        args.judge_model_name,
        args.judge_api_key,
        args.judge_max_workers,
        args.judge_timeout
    )

    # Evaluate
    results = []
    total_tokens = 0
    total_steps = 0
    correct_count = 0
    
    print("Evaluating responses...")
    for i, output in enumerate(tqdm(outputs)):
        generated_text = output.outputs[0].text
        # Use token_ids length if available, else approximate
        num_tokens = len(output.outputs[0].token_ids)
        total_tokens += num_tokens
        
        # Count steps
        step_count = count_steps(generated_text)
        total_steps += step_count
        
        row = df.iloc[i]
        question = row['query']
        ground_truth = row['ground_truth']
        
        predicted_answer = extract_answer(generated_text)
        
        score = judge.judge(question, predicted_answer, ground_truth)
        is_correct = (score == 1)
        if is_correct:
            correct_count += 1
            
        results.append({
            "prompt": row['prompt'],
            "query": question,
            "ground_truth": ground_truth,
            "generated_text": generated_text,
            "extracted_answer": predicted_answer,
            "is_correct": is_correct,
            "token_count": num_tokens,
            "step_count": step_count
        })

    # Metrics
    num_samples = len(results)
    accuracy = correct_count / num_samples if num_samples > 0 else 0
    avg_tokens = total_tokens / num_samples if num_samples > 0 else 0
    avg_steps = total_steps / num_samples if num_samples > 0 else 0
    
    print("="*40)
    print(f"Overall Accuracy: {accuracy:.4f} ({correct_count}/{num_samples})")
    print(f"Overall Avg Token Length: {avg_tokens:.2f}")
    print(f"Overall Avg Steps: {avg_steps:.2f}")
    print("="*40)

    # Hop Analysis
    categories = {
        1: {"count": 0, "correct": 0, "tokens": 0},
        2: {"count": 0, "correct": 0, "tokens": 0},
        3: {"count": 0, "correct": 0, "tokens": 0},
        4: {"count": 0, "correct": 0, "tokens": 0},
        5: {"count": 0, "correct": 0, "tokens": 0},
        "others": {"count": 0, "correct": 0, "tokens": 0}
    }

    for res in results:
        steps = res["step_count"]
        if steps in [1, 2, 3, 4, 5]:
            key = steps
        else:
            key = "others"
        
        categories[key]["count"] += 1
        if res["is_correct"]:
            categories[key]["correct"] += 1
        categories[key]["tokens"] += res["token_count"]

    print("\n" + "="*80)
    print(f"{'Category':<10} | {'Count':<10} | {'Ratio':<10} | {'Accuracy':<10} | {'Avg Tokens':<10}")
    print("-" * 80)
    
    if num_samples > 0:
        keys = [1, 2, 3, 4, 5, "others"]
        for k in keys:
            stats = categories[k]
            count = stats["count"]
            ratio = count / num_samples
            acc = stats["correct"] / count if count > 0 else 0.0
            avg_tok = stats["tokens"] / count if count > 0 else 0.0
            
            label = f"{k}-hop" if k != "others" else "others"
            print(f"{label:<10} | {count:<10} | {ratio:.2%}    | {acc:.4f}     | {avg_tok:.2f}")
    else:
        print("No results to analyze.")
    print("="*80 + "\n")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, "results.jsonl")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for res in results:
            f.write(json.dumps(res, ensure_ascii=False) + '\n')
    
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()

