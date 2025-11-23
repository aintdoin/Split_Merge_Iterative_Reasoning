import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple, Any

import matplotlib.pyplot as plt


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def align_samples(
    pre: List[Dict[str, Any]], post: List[Dict[str, Any]]
) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    # 优先使用 id 对齐；否则按顺序对齐
    pre_has_id = all("id" in r for r in pre)
    post_has_id = all("id" in r for r in post)

    if pre_has_id and post_has_id:
        pre_map = {r["id"]: r for r in pre}
        post_map = {r["id"]: r for r in post}
        if set(pre_map.keys()) != set(post_map.keys()):
            raise ValueError("两个JSONL的样本id集合不一致，无法对齐")
        aligned = [(pre_map[k], post_map[k]) for k in pre_map.keys()]
        return aligned

    if len(pre) != len(post):
        raise ValueError("两个JSONL行数不同且无法通过id对齐")
    return list(zip(pre, post))


def compute_transition(
    aligned: List[Tuple[Dict[str, Any], Dict[str, Any]]],
    answerable_value: bool,
    from_classes: List[int],
    to_classes: List[int],
) -> Tuple[Dict[int, Dict[int, int]], Dict[int, int], int, int]:
    # 返回：矩阵计数、每行总数、忽略计数、总样本数
    matrix: Dict[int, Dict[int, int]] = {fc: {tc: 0 for tc in to_classes} for fc in from_classes}
    row_totals: Dict[int, int] = {fc: 0 for fc in from_classes}
    ignored = 0
    total = 0

    for pre_rec, post_rec in aligned:
        if bool(pre_rec.get("answerable", False)) != answerable_value:
            continue

        total += 1
        pre_reward = int(pre_rec.get("reward"))
        post_reward = int(post_rec.get("reward"))

        if pre_reward not in from_classes:
            ignored += 1
            continue
        if post_reward not in to_classes:
            # 超出定义的目标类，忽略但计入行总数以避免偏差吗？
            # 这里选择忽略该样本，并不计入行总数。
            ignored += 1
            continue

        matrix[pre_reward][post_reward] += 1
        row_totals[pre_reward] += 1

    return matrix, row_totals, ignored, total


def format_summary(
    title: str,
    matrix: Dict[int, Dict[int, int]],
    row_totals: Dict[int, int],
    from_classes: List[int],
    to_classes: List[int],
    ignored: int,
    total: int,
) -> str:
    lines: List[str] = []
    lines.append(title)
    lines.append(f"样本总数(筛后)：{sum(row_totals.values())} / 原始同类样本：{total}，被忽略：{ignored}")
    header = ["from \\ to"] + [str(tc) for tc in to_classes]
    lines.append("\t".join(header))
    for fc in from_classes:
        row = [str(fc)]
        denom = row_totals.get(fc, 0) or 1
        for tc in to_classes:
            count = matrix[fc][tc]
            rate = count / denom
            row.append(f"{count} ({rate:.1%})")
        lines.append("\t".join(row))
    return "\n".join(lines)


def plot_heatmap(
    matrix: Dict[int, Dict[int, int]],
    row_totals: Dict[int, int],
    from_classes: List[int],
    to_classes: List[int],
    title: str,
    out_path: str,
):
    data = [[matrix[fc][tc] for tc in to_classes] for fc in from_classes]

    fig, ax = plt.subplots(figsize=(1.8 * len(to_classes) + 2, 1.6 * len(from_classes) + 2))
    im = ax.imshow(data, cmap="Blues", aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("Post-training reward")
    ax.set_ylabel("Pre-training reward")
    ax.set_xticks(range(len(to_classes)), labels=[str(x) for x in to_classes])
    ax.set_yticks(range(len(from_classes)), labels=[str(x) for x in from_classes])

    for i, fc in enumerate(from_classes):
        denom = row_totals.get(fc, 0) or 1
        for j, tc in enumerate(to_classes):
            c = matrix[fc][tc]
            rate = c / denom if denom else 0.0
            ax.text(j, i, f"{c}\n({rate:.1%})", ha="center", va="center", color="black")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Count")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="比较训练前后 reward 转移并可视化")
    parser.add_argument("pre_jsonl", type=str, help="训练前推理结果 JSONL")
    parser.add_argument("post_jsonl", type=str, help="训练后推理结果 JSONL")
    parser.add_argument(
        "--output_dir", type=str, default="output_pngs", help="输出目录，用于保存热力图"
    )
    args = parser.parse_args()

    pre = load_jsonl(args.pre_jsonl)
    post = load_jsonl(args.post_jsonl)
    aligned = align_samples(pre, post)

    os.makedirs(args.output_dir, exist_ok=True)

    # 检查并生成answerable=true的热力图（如果有样本）
    true_from = [-1, 0, 1]
    true_to = [-1, 0, 1]
    t_matrix, t_rows, t_ignored, t_total = compute_transition(
        aligned, True, true_from, true_to
    )
    
    # 只有当有实际样本（筛后总数大于0）时才生成热力图
    if sum(t_rows.values()) > 0:
        true_png = os.path.join(args.output_dir, "reward_transition_answerable_true.png")
        plot_heatmap(
            t_matrix,
            t_rows,
            true_from,
            true_to,
            "Reward transition (answerable=true)",
            true_png,
        )

    # 检查并生成answerable=false的热力图（如果有样本）
    false_from = [-1, 1]
    false_to = [-1, 1]
    f_matrix, f_rows, f_ignored, f_total = compute_transition(
        aligned, False, false_from, false_to
    )
    
    # 只有当有实际样本（筛后总数大于0）时才生成热力图
    if sum(f_rows.values()) > 0:
        false_png = os.path.join(args.output_dir, "reward_transition_answerable_false.png")
        plot_heatmap(
            f_matrix,
            f_rows,
            false_from,
            false_to,
            "Reward transition (answerable=false)",
            false_png,
        )


if __name__ == "__main__":
    main()

#python3 scripts/compare.py inference/inference_results/llama-idk/2wikimultihop.jsonl inference/inference_results/llama-true/2wikimultihop.jsonl --output_dir output_pngs
