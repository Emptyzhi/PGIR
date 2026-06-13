# PGIR

PGIR is a provenance-guided runtime repair method for diagnosing propagated
agent failures and rewriting the smallest structurally localizable repair
region.

The current implementation focuses on:

- runtime contract synthesis and verification;
- provenance and taint propagation;
- responsible ancestor frontier localization;
- deferred repair at dependency, fan-in, fan-out, plan, and final boundaries;
- interface-preserving local graph rewrites;
- structural escalation to global replanning only when no strict local repair
  region exists.

## Repository Layout

- `disasterbench/`: the main PGIR planning and repair harness.
- `integrations/wildclaw/`: PGIR runtime-controller overlay for WildClawBench.
- `docs/PGIR_METHOD.md`: current method design and research framing.

Benchmark datasets, Docker workspaces, API keys, logs, and generated results are
intentionally excluded.

## DisasterBench Quick Start

Clone DisasterBench into `third_party/DisasterBench_Open`, or set
`DISASTERBENCH_ROOT` to its absolute path.

Set runtime credentials:

```powershell
$env:DEEPSEEK_API_KEY = "..."
$env:TAVILY_API_KEY = "..."
```

Run the structural test without external API calls:

```powershell
python disasterbench/smoke_frontier_structural.py
```

Run a configured experiment phase:

```powershell
cd disasterbench
python main.py --list-phases
python main.py --phase local_graph_rewrite_10 --preflight
python main.py --phase local_graph_rewrite_10
```

`scorer.py` loads the official DisasterBench gold plan only after execution for
final evaluation. Gold data must never be used by PGIR contracts, diagnosis,
repair prompts, operator routing, or candidate selection.

## WildClaw Integration

`integrations/wildclaw/` is an overlay for an official WildClawBench checkout.
Copy its `src/` and `tests/` directories over the matching paths in
WildClawBench, then run the benchmark's test suite.

The overlay contains the runtime contract tree, provenance graph, repair
frontier controller, OpenClaw bridge, and component ablations.

## Versioning Policy

All future PGIR changes should be made in this repository and committed before
experiments are run. Use one focused commit per method change so experiments
can be traced to an exact revision and reverted with Git.

Useful commands:

```powershell
git status
git log --oneline --decorate -10
git switch -c codex/<change-name>
git revert <commit>
```

## Current Status

The runtime supports interface-preserving local graph rewrites that may add,
remove, merge, or split nodes inside a repair region while preserving the
semantics and dependencies of outside nodes. Structural tests pass. A targeted
10-task DisasterBench probe confirmed that the execution mechanism works, while
repair proposal quality and semantic verification remain the main bottlenecks.
