#!/usr/bin/env python
"""
将单行JSON文件转换为JSONL格式（每个样本独占一行）
"""

import json
import argparse
import os

def convert_json_to_jsonl(input_path, output_path):
    """
    读取单行JSON文件，并将每个样本写入单独的行
    
    参数:
        input_path: 输入JSON文件路径
        output_path: 输出JSONL文件路径
    """
    print(f"正在读取文件: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 确保数据是列表
    if not isinstance(data, list):
        raise ValueError("输入JSON文件必须包含JSON对象的数组/列表")
    
    print(f"共读取 {len(data)} 个样本")
    
    # 创建输出目录（如果不存在）
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # 写入JSONL文件，每行一个样本
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"转换完成！已生成JSONL文件: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将JSON文件转换为JSONL格式")
    parser.add_argument('--input', '-i', required=True, help="输入JSON文件路径")
    parser.add_argument('--output', '-o', required=True, help="输出JSONL文件路径")
    
    args = parser.parse_args()
    convert_json_to_jsonl(args.input, args.output)


