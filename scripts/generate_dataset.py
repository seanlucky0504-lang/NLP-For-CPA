import argparse
import json
from pathlib import Path

from src.teacher.agents import DatasetSynthesizer, PlannerAgent, ReviewerAgent, WriterAgent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量合成/蒸馏 CPA 教学数据并落地 JSON/JSONL")
    parser.add_argument("--topic", required=True, help="CPA 主题，如“财务成本管理”")
    parser.add_argument("--num-questions", type=int, default=200, help="目标样本数量")
    parser.add_argument(
        "--min-score",
        type=float,
        default=7.0,
        help="Reviewer 评分阈值，低于阈值的样本会被丢弃（蒸馏过滤）",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=None,
        help="最多尝试生成多少条（避免无限循环）；默认= num_questions * 3",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/generated"),
        help="输出目录（与 --output 二选一，若同时提供以 --output 为准）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="输出文件路径（优先级最高），适合断点续写同一个 JSONL",
    )
    parser.add_argument("--jsonl", action="store_true", help="以 JSONL 而非 JSON 列表保存")
    parser.add_argument(
        "--flush-every",
        type=int,
        default=50,
        help="每多少条追加落盘一次，避免长跑丢失（仅 JSONL 有效）",
    )
    parser.add_argument(
        "--start-id",
        type=int,
        default=1,
        help="起始样本 ID（用于断点续生，如已有 100 条则设为 101）",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="追加写入现有 JSONL（仅在 --jsonl 时有效；若文件存在且 --start-id 未指定，将自动从末尾 ID+1 续写）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    planner = PlannerAgent()
    writer = WriterAgent()
    reviewer = ReviewerAgent()
    synth = DatasetSynthesizer(planner, writer, reviewer)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    topic_slug = args.topic.replace(" ", "_")
    suffix = "jsonl" if args.jsonl else "json"
    output_path = (
        args.output
        if args.output
        else args.output_dir / f"{topic_slug}_teacher_{args.num_questions}.{suffix}"
    )

    if args.append and not args.jsonl:
        raise SystemExit("追加仅支持 JSONL 模式，请添加 --jsonl。")

    existing_count = 0
    if args.append and output_path.exists():
        with output_path.open("r", encoding="utf-8") as f:
            existing_count = sum(1 for _ in f)
        if args.start_id == 1:
            args.start_id = existing_count + 1
    else:
        # 清空旧文件再写
        output_path.write_text("", encoding="utf-8")

    def flush_chunk(chunk):
        with output_path.open("a", encoding="utf-8") as f:
            if args.jsonl:
                for item in chunk:
                    f.write(json.dumps(item.__dict__, ensure_ascii=False) + "\n")
            else:
                # 对 JSON 模式，先累积到内存，最后再重写；这里先写占位，后续覆盖
                pass

    dataset = synth.build(
        topic=args.topic,
        num_questions=args.num_questions,
        min_score=args.min_score,
        max_attempts=args.max_attempts,
        flush_every=args.flush_every if args.jsonl else None,
        flush_callback=flush_chunk if args.jsonl else None,
        start_id=args.start_id,
    )

    # 如果用户选择 JSON，最终一次性写为列表；JSONL 已经在过程中追加写入
    if args.jsonl:
        final_path = output_path
    else:
        output_path.write_text(
            json.dumps([item.__dict__ for item in dataset], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        final_path = output_path

    kept_ratio = len(dataset) / max(len(dataset), args.num_questions) * 100
    print(
        f"Saved {len(dataset)} samples to {final_path} "
        f"(start_id={args.start_id}, existing={existing_count}, kept ~{kept_ratio:.1f}% after score filter)"
    )


if __name__ == "__main__":
    main()
