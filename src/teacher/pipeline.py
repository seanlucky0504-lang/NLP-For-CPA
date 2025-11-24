import argparse
import json
from pathlib import Path

from .agents import DatasetSynthesizer, PlannerAgent, ReviewerAgent, WriterAgent


def run(topic: str, num_questions: int, output_path: Path) -> None:
    planner = PlannerAgent()
    writer = WriterAgent()
    reviewer = ReviewerAgent()
    synth = DatasetSynthesizer(planner, writer, reviewer)

    dataset = synth.build(topic=topic, num_questions=num_questions)
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
    parser.add_argument("--num-questions", type=int, default=5, help="Number of QA pairs")
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
