import argparse
import json
from pathlib import Path

from .agents import DatasetSynthesizer, PlannerAgent, ReviewerAgent, WriterAgent


def run(
    topic: str,
    num_questions: int,
    output_path: Path,
    min_score: float | None = None,
    max_attempts: int | None = None,
    use_outline: bool = False,
) -> None:
    planner = PlannerAgent()
    writer = WriterAgent()
    reviewer = ReviewerAgent()
    synth = DatasetSynthesizer(planner, writer, reviewer)

    dataset = synth.build(
        topic=topic,
        num_questions=num_questions,
        min_score=min_score,
        max_attempts=max_attempts,
        use_outline=use_outline,
    )
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
    parser.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="Optional reviewer score threshold; samples below are dropped",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=None,
        help="Safety cap on generation attempts (default: num_questions*3)",
    )
    parser.add_argument(
        "--use-outline",
        action="store_true",
        help="If set, will generate multi-point outline and teaching notes (slower, richer). Default is off for faster QA-only generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        topic=args.topic,
        num_questions=args.num_questions,
        output_path=args.output,
        min_score=args.min_score,
        max_attempts=args.max_attempts,
        use_outline=args.use_outline,
    )


if __name__ == "__main__":
    main()
