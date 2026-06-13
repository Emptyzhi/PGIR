"""Summarize the selective-repair diagnostic experiment."""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def mean(records, key):
    values = [float(record.get(key, 0.0) or 0.0) for record in records]
    return sum(values) / len(values) if values else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--phase", default="selective_repair_diagnostic_10")
    args = parser.parse_args()

    payload = json.loads(args.results.read_text(encoding="utf-8"))
    grouped = defaultdict(list)
    for record in payload[args.phase]:
        if "error" not in record:
            grouped[record["condition"]].append(record)

    print("| Condition | N | Primary | Tokens | Re-executed | Stopped | Fallbacks |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for condition, records in sorted(grouped.items()):
        print(
            f"| {condition} | {len(records)} | {mean(records, 'primary_metric'):.4f} "
            f"| {mean(records, 'total_repair_tokens'):.1f} "
            f"| {mean(records, 'reexecuted_steps'):.2f} "
            f"| {sum(bool(r.get('stopped_at_unrepaired_boundary')) for r in records)} "
            f"| {sum(int(r.get('selective_fallbacks', 0) or 0) for r in records)} |"
        )


if __name__ == "__main__":
    main()
