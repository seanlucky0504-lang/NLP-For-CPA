import argparse
import json
from pathlib import Path

from .agents import DatasetSynthesizer, PlannerAgent, ReviewerAgent, WriterAgent


def run(topic: str, num_questions: int, output_path: Path) -> None:
    planner = PlannerAgent()
    writer = WriterAgent()
    reviewer = ReviewerAgent()
    synth = DatasetSynthesizer(planner, writer, reviewer)

    def log_progress(event: dict) -> None:
        stage = event.get("stage")
        if stage == "outline":
            print(f"构建大纲完成，共 {event.get('count', 0)} 节，开始生成样本…", flush=True)
        elif stage == "prep":
            print(
                f"展开要点 {event.get('slots', 0)} 条，轮询生成 {num_questions} 题目", flush=True
            )
        elif stage == "sample":
            done = event.get("completed", 0)
            total = event.get("total", num_questions)
            eta = event.get("eta_seconds")
            eta_hint = f"，预计剩余 {eta:.1f}s" if eta is not None else ""
            print(
                f"[{done}/{total}] {event.get('section')} · 变体{event.get('variant')} · {event.get('difficulty')}" + eta_hint,
                flush=True,
            )

    dataset = synth.build(topic=topic, num_questions=num_questions, progress_callback=log_progress)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps([item.__dict__ for item in dataset], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved {len(dataset)} samples to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate CPA multi-agent Q&A dataset using DeepSeek via LangChain",
    )
    parser.add_argument("--topic", required=True, help="CPA subject, e.g., 财务成本管理")
    parser.add_argument(
        "--num-questions",
        type=int,
        default=200,
        help="Number of QA pairs (large outlines will be cycled to reach目标数量)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/teacher_generated.json"),
        help="Where to save the synthesized dataset",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(topic=args.topic, num_questions=args.num_questions, output_path=args.output)


if __name__ == "__main__":
    main()
