"""Summarize the fixed-task baseline fairness experiment."""
import argparse
import json
from collections import defaultdict


def mean(values):
    return sum(values) / len(values) if values else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results")
    parser.add_argument("--phase", default="baseline_fairness_10")
    args = parser.parse_args()
    payload = json.load(open(args.results, "r", encoding="utf-8"))
    records = [record for record in payload[args.phase] if "error" not in record]
    by_condition = defaultdict(list)
    for record in records:
        by_condition[record["condition"]].append(record)

    print("| condition | n | primary | tokens | LLM calls | search calls | reexecuted | contract pass |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for condition, items in sorted(
        by_condition.items(),
        key=lambda item: mean([record["primary_metric"] for record in item[1]]),
        reverse=True,
    ):
        print(
            f"| {condition} | {len(items)} | "
            f"{mean([record['primary_metric'] for record in items]):.4f} | "
            f"{mean([record.get('total_repair_tokens', 0) for record in items]):.1f} | "
            f"{mean([record.get('llm_calls', 0) for record in items]):.2f} | "
            f"{mean([record.get('search_calls', 0) for record in items]):.2f} | "
            f"{mean([record.get('reexecuted_steps', 0) for record in items]):.2f} | "
            f"{mean([record.get('contract_pass_rate', 0) for record in items]):.4f} |"
        )
    audit = payload.get("baseline_fairness_audit", {}).get("phases", {}).get(args.phase, {})
    print(f"\nFairness audit pass: {audit.get('pass', False)}")
    print(f"Groups checked: {audit.get('groups_checked', 0)}")


if __name__ == "__main__":
    main()
