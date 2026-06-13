"""PGIR experiment entry point."""

import argparse
import sys

from config import Config
from pgir_harness import ExperimentHarness


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PGIR experiment phases.")
    parser.add_argument(
        "--phase",
        action="append",
        dest="phases",
        help="Phase to run. May be passed multiple times. Defaults to config.execution_order.",
    )
    parser.add_argument(
        "--list-phases",
        action="store_true",
        help="List configured phases and exit.",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Validate phase datasets, conditions, and model keys without LLM calls.",
    )
    parser.add_argument(
        "--check-models",
        action="store_true",
        help="Probe configured model providers with tiny real LLM calls.",
    )
    args = parser.parse_args()
    try:
        config = Config()
    except RuntimeError as exc:
        print(f"Configuration error: {exc}")
        sys.exit(1)

    harness = ExperimentHarness(config)
    if args.list_phases:
        for phase_name in config.phases_definition:
            print(phase_name)
        return
    if args.preflight:
        harness.preflight(args.phases)
        return
    if args.check_models:
        from model_probe import main as probe_main

        raise SystemExit(probe_main([]))
    harness.run_phases(args.phases)


if __name__ == "__main__":
    main()
