import os
import re
import concurrent.futures
from typing import Optional

# Global singleton
_global_postprocessor = None


class AnswerPostProcessor:
    """
    Minimal answer post-processor focused on LLM-as-a-Judge.
    """
    
    def __init__(self):
        """
        Initialize with LLM Judge API configuration only.
        """
        # LLM Judge API configuration
        self.judge_api_base = os.environ.get('LLM_JUDGE_API_BASE', '').strip()
        self.judge_model_name = os.environ.get('LLM_JUDGE_MODEL_NAME', '').strip()
        self.judge_api_key = os.environ.get('LLM_JUDGE_API_KEY', '').strip()
        self.use_judge_api = bool(self.judge_api_base)
        
        # Check requests library availability
        try:
            import requests
            self.requests = requests
        except ImportError:
            self.requests = None
            if self.use_judge_api:
                print("⚠️  WARNING: requests library not available, LLM Judge will be disabled")
                self.use_judge_api = False
        
        # Concurrency settings
        self.max_workers = int(os.environ.get('LLM_JUDGE_MAX_WORKERS', '8'))
        # Increase timeout to avoid empty returns under load
        self.request_timeout = float(os.environ.get('LLM_JUDGE_TIMEOUT', '60'))
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) if self.use_judge_api else None
        
        # Debug logging
        print(f"[AnswerPostProcessor] Judge cfg -> base={bool(self.judge_api_base)}, model={'set' if self.judge_model_name else 'unset'}, workers={self.max_workers}, timeout={self.request_timeout}s")

    def process_answer(self, answer_text: str, expected_answer: str = None, question: str = None) -> str:
        """
        Apply only basic normalization (no LLM extraction).
        
        Args:
            answer_text: Raw answer from model
            expected_answer: Not used (kept for compatibility)
            question: Not used (kept for compatibility)
            
        Returns:
            Normalized answer string (basic normalization only)
        """
        if not answer_text:
            return ""
        
        # Basic normalization only
        answer = answer_text.strip()
        
        # Remove common prefixes
        prefixes = [
            "the answer is:",
            "the final answer is:",
            "answer:",
            "final answer:",
        ]
        answer_lower = answer.lower()
        for prefix in prefixes:
            if answer_lower.startswith(prefix):
                answer = answer[len(prefix):].strip()
                break
        
        # Remove surrounding quotes
        if (answer.startswith('"') and answer.endswith('"')) or \
           (answer.startswith("'") and answer.endswith("'")):
            answer = answer[1:-1].strip()
        
        return answer

    def process_answers_batch(self, answer_texts: list) -> list:
        """
        Apply basic normalization to a batch of answers.
        
        Args:
            answer_texts: List of raw answer strings from model
            
        Returns:
            List of normalized answer strings
        """
        return [self.process_answer(answer_text) for answer_text in answer_texts]

    def _rule_based_match(self, predicted: str, ground_truth: str) -> bool:
        """
        仅保留大小写无关的精确匹配。
        """
        if predicted is None or ground_truth is None:
            return False
        return str(predicted).strip().lower() == str(ground_truth).strip().lower()

    def _call_judge_api(self, prompt: str) -> str:
        """
        Internal helper to call the LLM Judge API.
        Returns the raw response text.
        """
        # Prefer chat-completions path for vLLM OpenAI server; fall back to completions
        base = self.judge_api_base.rstrip('/')
        chat_url = base + '/v1/chat/completions'
        comp_url = base + '/v1/completions'
        model_name = self.judge_model_name or 'llm-judge'
        headers = {'Content-Type': 'application/json'}
        if self.judge_api_key:
            headers['Authorization'] = f'Bearer {self.judge_api_key}'

        # Try chat endpoint first
        chat_payload = {
            'model': model_name,
            'messages': [
                {'role': 'system', 'content': 'You are a strict answer evaluator. Output only a single digit.'},
                {'role': 'user', 'content': prompt}
            ],
            'temperature': 0.0,
            'max_tokens': 5,
            'stream': False,
        }

        resp = self.requests.post(chat_url, json=chat_payload, headers=headers, timeout=self.request_timeout)
        if resp.status_code == 200:
            data = resp.json()
            text = ''
            try:
                text = data['choices'][0]['message']['content']
            except Exception:
                text = ''
            if text and text.strip():
                return text.strip()

        # Fallback to completions endpoint
        comp_payload = {
            'model': model_name,
            'prompt': prompt,
            'temperature': 0.0,
            'max_tokens': 8,
            'stream': False,
            'stop': ['\n']
        }
        resp2 = self.requests.post(comp_url, json=comp_payload, headers=headers, timeout=self.request_timeout)
        if resp2.status_code != 200:
            raise Exception(f"API error: {resp2.status_code}, {resp2.text[:200]}")
        data2 = resp2.json()
        text2 = data2.get('choices', [{}])[0].get('text', '').strip()
        # Ensure non-empty return to avoid downstream ambiguity (default to '-1')
        return text2 if text2 else '-1'

    def _extract_final_answer(self, text: str) -> str:
        """
        Extract final answer from <answer>...</answer> tags.
        If tags are not present, return the original text.
        """
        if not text:
            return ""
        
        # Extract text between <answer> and </answer> tags
        match = re.search(r'<answer>\s*(.*?)\s*</answer>', text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        # If no tags found, return original text
        return text.strip()
    
    def _clean_question(self, question: str) -> str:
        """
        No preprocessing needed for question - return as-is.
        """
        return question if question else ""
    
    def _is_idk_response(self, text: str) -> bool:
        """
        严格的 IDK 精确匹配（仅大小写差异）。
        """
        if not text:
            return False
        
        text_lower = text.strip().lower()
        
        # IDK markers（精确匹配）
        idk_markers = [
            "i don't know", "i dont know",
            "insufficient information", 
            "unknown"
        ]
        return any(text_lower == marker for marker in idk_markers)

    def judge_answer_correctness(self, predicted_answer: str, ground_truth_answer: str, question: str = None, answerable: Optional[bool] = None) -> int:
        """
        精简后的三步判断：
        1) 与 GT 大小写无关的精确匹配 -> 返回 1；
        2) 与 IDK 列表大小写无关的精确匹配 -> answerable=True 返回 0；answerable=False 返回 1；answerable=None 返回 0；
        3) 其余交给 LLM-as-a-Judge（统一使用提供的 prompt，输出 -1 或 1）。
        """
        # Preprocess inputs - extract final answer and clean question
        predicted_answer = self._extract_final_answer(predicted_answer) if predicted_answer else ""
        question = self._clean_question(question) if question else None
        
        # STAGE 1: Rule-based matching (high-confidence correct cases)
        # This stage can ONLY return True (confident match) or False (uncertain, needs LLM)
        if self._rule_based_match(predicted_answer, ground_truth_answer):
            # High-confidence match found via rule-based methods
            # For answerable questions, this is definitely correct (score=1)
            # For unanswerable questions, matching GT also means correct (score=1)
            # Note: We don't need to check answerable here because a match is a match
            return 1
        
        # STAGE 2: Rule-based IDK detection (NEW - critical for correctness!)
        # This prevents LLM parsing errors on "I don't know" responses
        if self._is_idk_response(predicted_answer):
            if answerable is True:
                return 0
            if answerable is False:
                return 1
            return 0
        
        # STAGE 3: LLM Judge (uncertain cases only)
        # Check if we have API configured
        if not self.use_judge_api:
            return -1
        
        # Build judge prompt（统一版本）
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
You should make the judgment based on provided examples.

Examples: 
Question: When did the director of film Lord Richard In The Pantry die?
Ground Truth: 7 January 1984
Prediction: January 7, 1984
Output: 1

Question: Who is older, Charles Badham or Médéric De Vasselot De Régné?
Ground Truth: Charles Badham
Prediction: Médéric De Vasselot De Régné
Output: -1

        Question: {question}
        Ground Truth: {ground_truth_answer}
        Prediction: {predicted_answer}
        Output: """
        # Call judge API (using thread pool for async execution)
        try:
            # concise call log
            
            # Submit to thread pool and wait for result
            future = self.executor.submit(self._call_judge_api, prompt)
            result = future.result(timeout=self.request_timeout + 5)  # Add buffer to timeout
            
            # print concise result head
            
            # Parse result - should be single digit: -1 or 1
            result_clean = result.strip()
            
            # ROBUST parsing: handle various output formats
            # 1. Try exact match first (best case)
            if result_clean in ['-1', '1']:
                score = int(result_clean)
            else:
                # 2. Try to find digit at the START (avoid parsing list numbers like "1. ")
                import re
                # Match digit at start of string, possibly with whitespace
                match = re.match(r'^\s*(-?\d+)', result_clean)
                if match:
                    score = int(match.group(1))
                else:
                    # 3. Last resort: search anywhere (but this is risky)
                    match = re.search(r'-?\d+', result_clean)
                    if match:
                        score = int(match.group())
                    else:
                        raise ValueError(f"No number found in: {result}")
            
            # Validate score range
            if score == -1:
                return -1
            elif score == 1:
                return 1
            else:
                # Out of range, use fallback
                raise ValueError(f"Score {score} not in {{-1, 1}}")
                
        except Exception as e:
            return -1

    def judge_batch(self, items: list) -> list:
        """
        Batch judge multiple answers concurrently using thread pool.
        
        Args:
            items: List of dicts, each containing:
                - predicted_answer: str
                - ground_truth_answer: str
                - question: str (optional)
                - answerable: bool (optional)
        
        Returns:
            List of scores (int) in the same order as input items
        """
        if not self.use_judge_api or not self.executor:
            # Fallback to sequential processing
            return [
                self.judge_answer_correctness(
                    item['predicted_answer'],
                    item['ground_truth_answer'],
                    item.get('question'),
                    item.get('answerable')
                )
                for item in items
            ]
        
        # Submit all tasks to thread pool
        futures = []
        for item in items:
            future = self.executor.submit(
                self.judge_answer_correctness,
                item['predicted_answer'],
                item['ground_truth_answer'],
                item.get('question'),
                item.get('answerable')
            )
            futures.append(future)
        
        # Collect results in order
        results = []
        for future in futures:
            try:
                result = future.result(timeout=self.request_timeout + 10)
                results.append(result)
            except Exception as e:
                results.append(0)  # Default neutral score on failure
        
        return results

    def shutdown(self):
        """Clean up resources"""
        if self.executor:
            self.executor.shutdown(wait=False)


def get_postprocessor() -> AnswerPostProcessor:
    """Get or create the global post-processor instance"""
    global _global_postprocessor
    
    if _global_postprocessor is None:
        _global_postprocessor = AnswerPostProcessor()
    
    return _global_postprocessor