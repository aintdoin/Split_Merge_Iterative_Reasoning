#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import tempfile
import os

# 导入处理函数
from process_data_for_reasoning import extract_evidence_items

def test_serialization():
    """测试序号分配功能"""
    
    # 测试数据：包含换行符的字符串
    test_evidences = """['Leif Tilden is the director of 1 Mile to You.'
 'Leos Carax is the director of Mauvais Sang.'
 "Leif Tilden's country of citizenship is the United States of America."
 "Leos Carax's country of citizenship is France."]"""
    
    print("原始数据:")
    print(repr(test_evidences))
    print()
    
    # 使用extract_evidence_items函数处理
    evidence_items = extract_evidence_items(test_evidences)
    
    print("提取的证据项:")
    for i, item in enumerate(evidence_items):
        print(f"{i+1}. {repr(item)}")
    print()
    
    # 清理证据项
    cleaned_evidence_items = []
    for item in evidence_items:
        item = item.strip()
        # 移除单引号或双引号
        if (item.startswith("'") and item.endswith("'")) or (item.startswith('"') and item.endswith('"')):
            item = item[1:-1]
        # 移除列表标记
        if item.startswith('[') and item.endswith(']'):
            item = item[1:-1]
        cleaned_evidence_items.append(item.strip())
    
    print("清理后的证据项:")
    for i, item in enumerate(cleaned_evidence_items):
        print(f"{i+1}. {repr(item)}")
    print()
    
    # 构建推理步骤
    reasoning_lines = []
    for i, evidence in enumerate(cleaned_evidence_items):
        reasoning_lines.append(f"{i+1}. {evidence}")
    reasoning_text = "\n".join(reasoning_lines)
    
    # 构建完整的response
    answer = "no"
    response = f"""<think>
{reasoning_text}
</think>
<answer>{answer}</answer>"""
    
    print("生成的response:")
    print(response)
    print()
    
    # 预期结果
    expected_response = """<think>
1. Leif Tilden is the director of 1 Mile to You.
2. Leos Carax is the director of Mauvais Sang.
3. Leif Tilden's country of citizenship is the United States of America.
4. Leos Carax's country of citizenship is France.
</think>
<answer>no</answer>"""
    
    print("预期结果:")
    print(expected_response)
    print()
    
    # 比较结果
    if response == expected_response:
        print("✅ 测试通过！序号分配正确。")
    else:
        print("❌ 测试失败！序号分配不正确。")
        print("差异:")
        print("实际结果:")
        print(repr(response))
        print("预期结果:") 
        print(repr(expected_response))

if __name__ == "__main__":
    test_serialization()