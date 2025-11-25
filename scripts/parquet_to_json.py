#!/usr/bin/env python3
import argparse
import json
import os
import sys
from typing import Optional

try:
    import pyarrow.parquet as pq
except Exception as e:
    print("[ERROR] 需要安装 pyarrow 才能读取 parquet。请先运行: pip install pyarrow", file=sys.stderr)
    raise


def infer_output_path(input_path: str, output_path: Optional[str], jsonl: bool) -> str:
    if output_path:
        return output_path
    base, _ = os.path.splitext(input_path)
    return base + (".jsonl" if jsonl else ".json")


def convert_parquet(input_path: str,
                    output_path: str,
                    jsonl: bool = False,
                    limit: Optional[int] = None,
                    indent: Optional[int] = 2,
                    batch_size: int = 1000) -> int:
    """将 Parquet 流式转换为 JSON/JSONL。

    - JSONL: 每行一个样本，适合超大数据；
    - JSON: 输出为 JSON 数组，采用流式写入避免占用过多内存。
    返回写出的样本条数。
    """
    pf = pq.ParquetFile(input_path)
    num_written = 0

    # 确保目标目录存在
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as fout:
        if not jsonl:
            fout.write("[\n")

        first_record = True
        for rb in pf.iter_batches(batch_size=batch_size):
            # 将 RecordBatch 转为 Python 原生对象列表
            records = rb.to_pylist()
            for rec in records:
                if limit is not None and num_written >= limit:
                    break

                if jsonl:
                    fout.write(json.dumps(rec, ensure_ascii=False))
                    fout.write("\n")
                else:
                    if not first_record:
                        fout.write(",\n")
                    fout.write(json.dumps(rec, ensure_ascii=False, indent=indent))
                    first_record = False

                num_written += 1

            if limit is not None and num_written >= limit:
                break

        if not jsonl:
            fout.write("\n]\n")

    return num_written


def main():
    parser = argparse.ArgumentParser(description="Convert Parquet to JSON or JSONL (streaming, low memory)")
    parser.add_argument("input", help="输入的 .parquet 文件路径")
    parser.add_argument("--output", "-o", help="输出文件路径（默认与输入同名，后缀为 .json / .jsonl）")
    parser.add_argument("--jsonl", action="store_true", help="输出为 JSONL（每行一个 JSON）")
    parser.add_argument("--limit", type=int, default=None, help="最多导出多少条（默认导出全部）")
    parser.add_argument("--indent", type=int, default=2, help="JSON 缩进（仅对 .json 有效）")
    parser.add_argument("--batch-size", type=int, default=1000, help="读取批大小（影响内存占用与速度）")

    args = parser.parse_args()

    input_path = args.input
    if not os.path.isfile(input_path):
        print(f"[ERROR] 输入文件不存在: {input_path}", file=sys.stderr)
        sys.exit(1)

    output_path = infer_output_path(input_path, args.output, args.jsonl)

    try:
        num = convert_parquet(
            input_path=input_path,
            output_path=output_path,
            jsonl=args.jsonl,
            limit=args.limit,
            indent=args.indent,
            batch_size=args.batch_size,
        )
    except Exception as e:
        print(f"[ERROR] 转换失败: {e}", file=sys.stderr)
        sys.exit(2)

    print(f"[OK] 已写出 {num} 条到: {output_path}")


if __name__ == "__main__":
    main()


"""
# 转为 JSON（数组），自动推断输出路径
python scripts/parquet_to_json.py /path/to/file.parquet

# 转为 JSONL（每行一个样本）
python scripts/parquet_to_json.py /mnt/shared-storage-user/liyafu/runquan/Ineffective-Thinking-main/data/musique_ans/deepseek-r1-distill-qwen/train.parquet --jsonl

# 指定输出路径
python scripts/parquet_to_json.py /path/to/file.parquet -o /path/to/out.jsonl --jsonl

# 只导出前 N 条（快速查看）
python scripts/parquet_to_json.py /mnt/shared-storage-user/liyafu/runquan/SMIR/SFT/data/musique_train.parquet --jsonl --limit 5

# 控制 JSON 缩进（仅对 .json 生效）
python scripts/parquet_to_json.py /path/to/file.parquet --indent 2
"""