import argparse
import json
from pathlib import Path
from typing import Iterable, List

import evaluate


def load_json(path: Path) -> List[dict]:
    text = path.read_text(encoding="utf-8")
    if text.lstrip().startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def compute_metrics(references: Iterable[str], predictions: Iterable[str]) -> dict:
    refs = list(references)
    preds = list(predictions)

    bleu = evaluate.load("sacrebleu")
    bertscore = evaluate.load("bertscore")

    bleu_score = bleu.compute(predictions=preds, references=[[r] for r in refs])
    bert_score = bertscore.compute(
        predictions=preds,
        references=refs,
        lang="zh",
        model_type="bert-base-chinese",
    )
    return {
        "bleu": bleu_score["score"],
        "bertscore_precision": sum(bert_score["precision"]) / len(preds),
        "bertscore_recall": sum(bert_score["recall"]) / len(preds),
        "bertscore_f1": sum(bert_score["f1"]) / len(preds),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate student outputs against teacher references")
    parser.add_argument("--teacher", type=Path, required=True, help="Teacher JSON dataset")
    parser.add_argument("--student", type=Path, required=True, help="Student model outputs JSON")
    args = parser.parse_args()

    teacher = load_json(args.teacher)
    student = load_json(args.student)

    # assume same ordering by id
    teacher_by_id = {item["id"]: item for item in teacher}
    preds, refs = [], []
    for sample in student:
        ref = teacher_by_id.get(sample["id"], {})
        refs.append(ref.get("output", ""))
        preds.append(sample.get("output", ""))

    metrics = compute_metrics(references=refs, predictions=preds)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
