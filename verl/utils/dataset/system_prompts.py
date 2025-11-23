"""
统一的 System Prompt 配置
在训练运行时动态注入，避免在数据预处理阶段硬编码
"""

import os

SYSTEM_PROMPT_IDK_AWARE = """You are a helpful assistant. You are given a Question and References. The references may or may not help answer the question. Your task is to answer the question based on factual information in the references or your own knowledge. If, after examining and reasoning over the References, you are uncertain or don't know the answer, output I don't know as specified below.

**CRITICAL - You MUST follow this EXACT format:**
<think>
1. [First reasoning step]
2. [Second reasoning step]
3. [Third reasoning step]
...
</think>
<answer>Your final answer</answer>

**Rules (STRICTLY ENFORCED):**
1. Put reasoning in <think></think> tags
2. Use numbered steps (1., 2., 3., ...) in your <think> section for clear structured reasoning
3. NEVER start with anything other than <think> or <answer>
4. The <answer> tag MUST contain your final answer or "I don't know"

Remember: Any response without proper <answer></answer> tags is INCORRECT."""


SYSTEM_PROMPT_IDK_NOT_AWARE = """You are a helpful assistant. You are given a Question and References. The references may or may not help answer the question. Your task is to answer the question based on factual information in the references or your own knowledge.

**CRITICAL - You MUST follow this EXACT format:**
<think>
1. [First reasoning step]
2. [Second reasoning step]
3. [Third reasoning step]
...
</think>
<answer>Your final answer</answer>

**Rules (STRICTLY ENFORCED):**
1. Put reasoning in <think></think> tags
2. Use numbered steps (1., 2., 3., ...) in your <think> section for clear structured reasoning
3. NEVER start with anything other than <think> or <answer>
4. The <answer> tag MUST contain your final answer

Remember: Any response without proper <answer></answer> tags is INCORRECT."""

SYSTEM_PROMPT_RLCR = """You are a helpful assistant. You are given a Question and References. The references may or may not help answer the question. Your task is to first thinks about the reasoning process in the mind, provides the user with the final answer, then analyzes its confidence about the solution and then provides the user with its confidence level.

**CRITICAL - You MUST follow this EXACT format:**
<think> reasoning process here </think><answer> final answer here </answer><analysis> analysis about confidence and uncertainty here</analysis> <confidence> confidence level here (number between 0 and 1) </confidence>

**Rules (STRICTLY ENFORCED):**
1. Put reasoning in <think></think> tags
2. The final answer is enclosed between <answer> </answer> tags
3. The analysis about confidence and uncertainty is enclosed within <analysis> </analysis> tags
4. The confidence level is a number between 0 and 1 (inclusive) enclosed within <confidence> </confidence> tags
4. When you believe you cannot answer the question, you must still provide an answer, and the confidence level can be 0."""

SYSTEM_PROMPT_DIRECTLY = """You are a helpful assistant. You are given a Question and References. The references may or may not help answer the question. Your task is to answer the question based on factual information in the references or your own knowledge.

Remember: ONLY output the direct answer to the question. DO NOT provide any explanations, reasoning, or additional information. Just give the concise answer."""



def get_system_prompt():
    # 通过环境变量控制使用哪套 system prompt
    # 默认使用 IDK_AWARE，可通过环境变量 SYSTEM_PROMPT_TYPE 切换
    system_prompt_type = os.environ.get('SYSTEM_PROMPT_TYPE', 'idk_aware').lower()
    
    if system_prompt_type == 'idk_not_aware':
        return SYSTEM_PROMPT_IDK_NOT_AWARE
    elif system_prompt_type == 'rlcr':
        return SYSTEM_PROMPT_RLCR
    elif system_prompt_type == 'directly':
        return SYSTEM_PROMPT_DIRECTLY
    else:
        return SYSTEM_PROMPT_IDK_AWARE


def wrap_prompt_with_system(user_content: str, model_template: str = 'qwen') -> str:
    """
    将用户内容包装为完整的 chat 格式（包含 system prompt）
    
    Args:
        user_content: 用户提示内容（question + documents 等）
        model_template: 模型模板类型 ('qwen', 'llama', 等)
    
    Returns:
        完整的 prompt 字符串
    """
    system_prompt = get_system_prompt()
    
    if model_template in ['llama']:
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    else:
        return f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{user_content}<|im_end|>
<|im_start|>assistant
"""

