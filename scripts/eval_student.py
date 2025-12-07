import argparse
import json
from pathlib import Path
from typing import Iterable, List

from src.eval.evaluate import compute_metrics
from src.student.inference import chat, load_student


def load_records(path: Path) -> List[dict]:
    text = path.read_text(encoding="utf-8")
    if text.lstrip().startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate student answers and evaluate against teacher references")
    parser.add_argument("--teacher", type=Path, required=True, help="Teacher JSON/JSONL with id/input/output")
    parser.add_argument("--model", default="Qwen1.5-1.8B-Chat")
    parser.add_argument("--lora", type=Path, default=Path("outputs/student_lora"))
    parser.add_argument("--limit", type=int, default=200, help="Max samples to evaluate")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--pred-output",
        type=Path,
        default=Path("outputs/student_preds.json"),
        help="Where to store student generations",
    )
    args = parser.parse_args()

    teacher_records = load_records(args.teacher)[: args.limit]

    tokenizer, model = load_student(args.model, args.lora)
    preds: List[dict] = []
    for sample in teacher_records:
        answer = chat(
            tokenizer,
            model,
            sample["input"],
            max_new_tokens=args.max_new_tokens,
        )
        preds.append({"id": sample["id"], "input": sample["input"], "output": answer})

    args.pred_output.parent.mkdir(parents=True, exist_ok=True)
    args.pred_output.write_text(json.dumps(preds, ensure_ascii=False, indent=2), encoding="utf-8")

    metrics = compute_metrics(
        references=[item["output"] for item in teacher_records],
        predictions=[p["output"] for p in preds],
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

