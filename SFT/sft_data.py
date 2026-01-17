import os
import re
import json
import asyncio
import pandas as pd
import aiohttp
import argparse
from typing import List, Optional, Dict

# Regex patterns
THINK_PATTERN = re.compile(r'<think>(.*?)</think>', re.DOTALL)
ANSWER_PATTERN = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)
SPLIT_PATTERN = re.compile(r'\n(?=\d+\.)')

class ReasoningRefiner:
    def __init__(self, api_base: str, api_key: str, model_name: str, timeout: int = 60, max_workers: int = 8):
        self.api_base = api_base.strip().rstrip('/')
        # Handle /v1 suffix robustly: remove if present, then append standard path
        if self.api_base.endswith('/v1'):
            self.api_base = self.api_base[:-3]
            
        self.url = f"{self.api_base}/v1/chat/completions"
        self.api_key = api_key
        self.model_name = model_name
        self.timeout = timeout
        self.semaphore = asyncio.Semaphore(max_workers)
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        print(f"DEBUG: Initialized ReasoningRefiner with URL: {self.url} Timeout: {self.timeout}")


    async def _call_llm(self, messages: List[Dict[str, str]], temp: float = 0.0) -> str:
        async with self.semaphore:
            payload = {
                "model": self.model_name,
                "messages": messages,
                "temperature": temp,
                "max_tokens": 1024,
                "stream": False
            }
            async with aiohttp.ClientSession() as session:
                try:
                    async with session.post(self.url, headers=self.headers, json=payload, timeout=self.timeout) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            return data['choices'][0]['message']['content']
                        else:
                            error_text = await resp.text()
                            print(f"API Error {resp.status}: {error_text}")
                            return ""
                except Exception as e:
                    import traceback
                    print(f"Request failed: {e}")
                    traceback.print_exc()
                    return ""

    async def process_c3ot_segment(self, segment: str) -> Optional[str]:
        system_msg = "You are an expert logic judge."
        user_msg = (
            f"Review the following reasoning step. Decide if it is useful and necessary for the logical flow.\n"
            f"Step: {segment}\n"
            f"If it is useless, redundant, or empty, answer NO.\n"
            f"If it is useful, answer YES.\n"
            f"Reply with YES or NO only."
        )
        
        response = await self._call_llm([
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ])
        
        if "YES" in response.upper():
            return segment
        return None

    async def process_cosmo_pair(self, s1: str, s2: str) -> dict:
        instruction = (
            "Each step represents a **complete logical inference**. "
            "Do NOT split a single thought or evidence extraction into multiple tiny steps. "
            "Combine retrieving a document, quoting it, and deducing a fact into ONE single numbered step."
        )
        
        system_msg = "You are a logical editor. Output strictly in JSON."
        user_msg = (
            f"Reasoning Step 1: {s1}\n"
            f"Reasoning Step 2: {s2}\n\n"
            f"Rules:\n{instruction}\n\n"
            f"Task: Check if these two steps belong to the same logical inference.\n"
            f"1. If they are part of the same step, merge and refine them into one concise sentence.\n"
            f"2. If they are different steps, refine each of them separately to be clear and concise.\n\n"
            f"Output JSON format:\n"
            f"{{\n"
            f'  "decision": "merge" OR "split",\n'
            f'  "merged_text": "string (only if merge)",\n'
            f'  "refined_1": "string (only if split)",\n'
            f'  "refined_2": "string (only if split)"\n'
            f"}}"
        )

        response = await self._call_llm([
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ])

        try:
            # Simple cleanup for json parsing
            clean_json = response.strip()
            if clean_json.startswith("```json"):
                clean_json = clean_json.replace("```json", "").replace("```", "")
            return json.loads(clean_json)
        except:
            # Fallback on failure: treat as split, keep original
            return {"decision": "split", "refined_1": s1, "refined_2": s2}

    async def process_cosmo_force_merge(self, reasoning_text: str, hops: int) -> Optional[List[str]]:
        """
        Force compress the whole reasoning into exactly `hops` numbered steps.
        Return a list of step strings (without numeric prefixes) on success, else None.
        """
        if hops not in (2, 3, 4):
            raise ValueError(f"cosmo_hops must be 2/3/4, got {hops}")

        instruction = (
            "Each step must represent a **complete logical inference**. "
            "Do NOT split one inference into multiple tiny steps. "
            "You may merge adjacent steps and remove redundancy, but do NOT add new facts. "
            f"Output MUST contain exactly {hops} steps."
        )

        system_msg = "You are a logical editor. Output strictly in JSON."
        user_msg = (
            f"Input reasoning steps (may be verbose):\n{reasoning_text}\n\n"
            f"Rules:\n{instruction}\n\n"
            "Task: Rewrite/compress the reasoning into exactly N steps (N given above).\n\n"
            "Output JSON format:\n"
            "{\n"
            '  "steps": ["step1", "step2", "..."]\n'
            "}\n"
            "Constraints:\n"
            f"- steps MUST be a JSON array of length {hops}\n"
            "- Each element is a concise sentence/paragraph WITHOUT numeric prefix\n"
        )

        response = await self._call_llm([
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ])

        def _strip_fences(text: str) -> str:
            t = text.strip()
            if t.startswith("```"):
                t = re.sub(r"^```[a-zA-Z0-9]*\n?", "", t)
                t = t.replace("```", "")
            return t.strip()

        try:
            clean_json = _strip_fences(response)
            data = json.loads(clean_json)
            steps = data.get("steps")
            if isinstance(steps, list):
                norm = []
                for s in steps:
                    if not isinstance(s, str):
                        continue
                    # Remove accidental numbering like "1. ..."
                    s2 = re.sub(r'^\s*\d+\.\s*', '', s).strip()
                    if s2:
                        norm.append(s2)
                if len(norm) == hops:
                    return norm
        except Exception:
            pass

        # Fallback: try to parse numbered lines from raw response
        lines = [ln.strip() for ln in response.splitlines() if ln.strip()]
        parsed = []
        for ln in lines:
            m = re.match(r'^\s*(\d+)\.\s*(.*)$', ln)
            if m and m.group(2).strip():
                parsed.append(m.group(2).strip())
        if len(parsed) == hops:
            return parsed

        return None

    async def run_c3ot_pipeline(self, thoughts: str) -> str:
        segments = [s.strip() for s in re.split(SPLIT_PATTERN, thoughts) if s.strip()]
        tasks = [self.process_c3ot_segment(seg) for seg in segments]
        results = await asyncio.gather(*tasks)
        
        valid_steps = [r for r in results if r is not None]
        
        # Reformat with numbers
        formatted_steps = []
        for i, step in enumerate(valid_steps):
            # Remove existing number prefix if present to normalize
            clean_step = re.sub(r'^\d+\.\s*', '', step)
            formatted_steps.append(f"{i+1}. {clean_step}")
            
        return "\n".join(formatted_steps)

    async def run_cosmo_pipeline(self, thoughts: str, cosmo_hops: Optional[int] = None) -> str:
        segments = [s.strip() for s in re.split(SPLIT_PATTERN, thoughts) if s.strip()]
        if not segments:
            return ""
            
        # Clean initial segments of numbers
        segments = [re.sub(r'^\d+\.\s*', '', s) for s in segments]
        
        final_segments = []
        if not segments:
            return ""
            
        current_segment = segments[0]
        
        for next_segment in segments[1:]:
            result = await self.process_cosmo_pair(current_segment, next_segment)
            
            if result.get("decision") == "merge":
                current_segment = result.get("merged_text", current_segment + " " + next_segment)
            else:
                # Split: finalize current, set next as new current
                refined_current = result.get("refined_1", current_segment)
                final_segments.append(refined_current)
                current_segment = result.get("refined_2", next_segment)
                
        # Append the last remaining segment
        final_segments.append(current_segment)

        # Optional: enforce fixed hops=n (n in 2/3/4) with a single extra Judge call
        if cosmo_hops is None:
            # Allow env-var override for convenience in scripts
            env_hops = os.environ.get("COSMO_HOPS", "").strip()
            if env_hops:
                try:
                    cosmo_hops = int(env_hops)
                except Exception:
                    cosmo_hops = None

        if cosmo_hops is not None:
            if cosmo_hops not in (2, 3, 4):
                raise ValueError(f"--cosmo_hops must be 2/3/4, got {cosmo_hops}")

            # If we still have too many segments, force-merge the whole reasoning into exactly n steps
            # (threshold uses n+1 to allow a final wrap-up step in some datasets)
            if len(final_segments) > (cosmo_hops + 1):
                current_reasoning_text = "\n".join([f"{i+1}. {s}" for i, s in enumerate(final_segments)])
                forced = await self.process_cosmo_force_merge(current_reasoning_text, cosmo_hops)
                if forced:
                    return "\n".join([f"{i+1}. {s}" for i, s in enumerate(forced)])

        # Default format (after one top-to-bottom pass)
        return "\n".join([f"{i+1}. {s}" for i, s in enumerate(final_segments)])

async def process_row(refiner: ReasoningRefiner, row: pd.Series, method: str, cosmo_hops: Optional[int]) -> tuple[dict, str]:
    generated_text = row.get("generated_text", "")
    prompt_text = row.get("prompt", "")
    
    # Extract
    think_match = THINK_PATTERN.search(generated_text)
    answer_match = ANSWER_PATTERN.search(generated_text)
    
    original_think = think_match.group(1).strip() if think_match else ""
    original_answer = answer_match.group(1).strip() if answer_match else row.get("extracted_answer", "")
    
    if not original_think:
        # Fallback if no thought found
        return {"prompt": prompt_text, "response": generated_text}, "skipped_no_think"
        
    new_think = ""
    if method == "C3oT":
        new_think = await refiner.run_c3ot_pipeline(original_think)
    elif method == "cosmo":
        new_think = await refiner.run_cosmo_pipeline(original_think, cosmo_hops=cosmo_hops)
        
    # Reassemble
    new_response = f"<think>\n{new_think}\n</think>\n<answer>{original_answer}</answer>"
    
    status = "success" if new_think.strip() else "processed_empty"

    return {
        "prompt": prompt_text,
        "response": new_response
    }, status

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True, choices=["cosmo", "C3oT"])
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of samples to process")
    parser.add_argument(
        "--cosmo_hops",
        type=int,
        default=None,
        choices=[2, 3, 4],
        help="Only for --method cosmo. Fixed hops n in {2,3,4}. If final segments > n+1, force merge into exactly n steps."
    )
    args = parser.parse_args()
    
    # Environment Variables
    api_base = os.environ.get("LLM_JUDGE_API_BASE", "http://localhost:8000/v1").strip()
    api_key = os.environ.get("LLM_JUDGE_API_KEY", "EMPTY").strip()
    model_name = os.environ.get("LLM_JUDGE_MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct").strip()
    max_workers = int(os.environ.get("LLM_JUDGE_MAX_WORKERS", "8"))
    timeout = int(os.environ.get("LLM_JUDGE_TIMEOUT", "60"))
    
    print(f"Starting {args.method} processing...")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Judge Model: {model_name}")
    print(f"Timeout: {timeout}")
    
    refiner = ReasoningRefiner(api_base, api_key, model_name, timeout, max_workers)
    
    # Read Data
    # Using chunksize if file is huge, but for now loading all
    data = []
    with open(args.input, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    df = pd.DataFrame(data)
    print(f"Loaded {len(df)} samples.")

    if args.max_samples is not None and args.max_samples > 0:
        df = df.head(args.max_samples)
        print(f"Limiting to first {args.max_samples} samples.")
    
    # Process
    tasks = []
    skipped_incorrect = 0
    for _, row in df.iterrows():
        if args.method == "cosmo":
            # Only process correct samples for cosmo
            # Use strict boolean check or string 'true'
            val = row.get("is_correct", False)
            is_correct = False
            
            if isinstance(val, bool):
                is_correct = val
            elif isinstance(val, str):
                is_correct = (val.lower() == 'true')
            # If val is 1/0 (int)
            elif isinstance(val, (int, float)):
                is_correct = bool(val)
                
            if not is_correct:
                skipped_incorrect += 1
                continue

        tasks.append(process_row(refiner, row, args.method, args.cosmo_hops))
    
    # Run with progress bar logic could be added, but simple gather for now
    # To avoid OOM with too many tasks, maybe chunk?
    # Semaphore in class handles concurrency, but thousands of task objects might be heavy.
    # Let's chunk the gather.
    
    chunk_size = 50
    final_data = []
    status_counts = {"skipped_no_think": 0, "success": 0, "processed_empty": 0}
    
    total_chunks = (len(tasks) + chunk_size - 1) // chunk_size
    for i in range(0, len(tasks), chunk_size):
        chunk = tasks[i:i + chunk_size]
        print(f"Processing chunk {i//chunk_size + 1}/{total_chunks}...")
        chunk_results = await asyncio.gather(*chunk)
        
        for res_dict, status in chunk_results:
            final_data.append(res_dict)
            status_counts[status] = status_counts.get(status, 0) + 1
        
    # Save
    out_df = pd.DataFrame(final_data)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    out_df.to_parquet(args.output, index=False)
    
    print("-" * 30)
    print("Processing Summary:")
    if args.method == "cosmo":
        print(f"  - Skipped (incorrect source): {skipped_incorrect}")
    print(f"Total processed: {len(final_data)}")
    print(f"  - Successfully refined: {status_counts.get('success', 0)}")
    print(f"  - Skipped (no think block): {status_counts.get('skipped_no_think', 0)}")
    print(f"  - Empty result (filtered/error): {status_counts.get('processed_empty', 0)}")
    print("-" * 30)
    print("Done.")

if __name__ == "__main__":
    asyncio.run(main())

