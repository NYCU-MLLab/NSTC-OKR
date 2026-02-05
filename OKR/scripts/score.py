#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def _safe_str(val):
    if val is None:
        return ""
    return str(val).strip()


def load_predictions(path: Path):
    records = []
    parse_fail_count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                parse_fail_count += 1
    return records, parse_fail_count


def infer_correct(rec):
    if "correct" in rec:
        return bool(rec["correct"])
    pred_final = rec.get("pred_final", rec.get("pred_text", ""))
    gold_final = rec.get("gold_final", rec.get("gold", ""))
    return _safe_str(pred_final) == _safe_str(gold_final)


def main():
    parser = argparse.ArgumentParser(description="Score predictions.jsonl and output metrics.json")
    parser.add_argument("--predictions", required=True, help="Path to predictions.jsonl")
    parser.add_argument("--output", required=True, help="Path to metrics.json")
    parser.add_argument("--wrong-out", default=None, help="Optional path to wrong_cases.jsonl")
    args = parser.parse_args()

    pred_path = Path(args.predictions)
    out_path = Path(args.output)
    records, parse_fail_count = load_predictions(pred_path)

    correct = 0
    wrong_records = []
    for rec in records:
        is_correct = infer_correct(rec)
        if is_correct:
            correct += 1
        else:
            wrong_records.append(rec)

    num_examples = len(records)
    accuracy = (correct / num_examples) if num_examples else 0.0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "accuracy": round(accuracy, 6),
                "num_examples": num_examples,
                "parse_fail_count": parse_fail_count,
            },
            f,
            ensure_ascii=True,
            indent=2,
        )

    if args.wrong_out:
        wrong_path = Path(args.wrong_out)
        wrong_path.parent.mkdir(parents=True, exist_ok=True)
        with wrong_path.open("w", encoding="utf-8") as f:
            for rec in wrong_records:
                f.write(json.dumps(rec, ensure_ascii=True) + "\n")


if __name__ == "__main__":
    main()
