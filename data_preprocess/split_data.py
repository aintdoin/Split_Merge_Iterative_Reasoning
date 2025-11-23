#!/usr/bin/env python3

import pandas as pd
import argparse
import os
from tqdm import tqdm


def split_by_answerable(input_file, output_true_file, output_false_file):
    """
    Args:
        input_file: 输入的parquet文件路径
        output_true_file: answerable=True的输出文件路径
        output_false_file: answerable=False的输出文件路径
    """
    print(f"正在读取文件: {input_file}")
    df = pd.read_parquet(input_file)
    print(f"总共读取了 {len(df)} 条数据")
    
    # 检查extra_info列是否存在
    if 'extra_info' not in df.columns:
        raise ValueError("输入文件中没有'extra_info'列")
    
    # 初始化两个DataFrame
    true_data = []
    false_data = []
    
    print("正在按answerable字段分类数据...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        extra_info = row['extra_info']
        
        # 检查extra_info是否是字典类型
        if isinstance(extra_info, dict):
            answerable = extra_info.get('answerable', None)
        else:
            # 如果extra_info不是字典，尝试解析字符串
            try:
                import ast
                extra_info_dict = ast.literal_eval(extra_info)
                answerable = extra_info_dict.get('answerable', None)
            except:
                print(f"警告: 第{idx}行的extra_info无法解析: {extra_info}")
                continue
        
        # 根据answerable值分类
        if answerable is True:
            true_data.append(row)
        elif answerable is False:
            false_data.append(row)
        else:
            print(f"警告: 第{idx}行的answerable字段缺失或无效: {answerable}")
    
    # 创建DataFrame
    true_df = pd.DataFrame(true_data)
    false_df = pd.DataFrame(false_data)
    
    print(f"answerable=True 的数据: {len(true_df)} 条")
    print(f"answerable=False 的数据: {len(false_df)} 条")
    
    # 保存到文件
    print(f"正在保存answerable=True的数据到: {output_true_file}")
    true_df.to_parquet(output_true_file, index=False)
    
    print(f"正在保存answerable=False的数据到: {output_false_file}")
    false_df.to_parquet(output_false_file, index=False)
    
    print("处理完成!")


def main():
    parser = argparse.ArgumentParser(description="将文件按answerable字段分类")
    parser.add_argument("--input", type=str, required=True, help="输入的parquet文件路径")
    parser.add_argument("--output_true", type=str, default="ans_true.parquet", 
                        help="answerable=True的输出文件路径")
    parser.add_argument("--output_false", type=str, default="ans_false.parquet", 
                        help="answerable=False的输出文件路径")
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"输入文件不存在: {args.input}")
    
    # 创建输出目录（如果不存在）
    os.makedirs(os.path.dirname(os.path.abspath(args.output_true)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_false)), exist_ok=True)
    
    # 执行分类
    split_by_answerable(args.input, args.output_true, args.output_false)


if __name__ == "__main__":
    main()

