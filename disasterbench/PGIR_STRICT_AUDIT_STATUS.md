# PGIR Strict Implementation Audit

## Implemented Invariants

- Runtime dependencies are inferred from explicit dependencies,
  `dependence_content`, generated-output references, and labeled textual
  references.
- Repair is deferred to plan, dependency-consumption, fan-in, fan-out, and
  final-commit boundaries.
- Multiple blocking violations are collected into a responsible ancestor
  frontier and repaired jointly.
- Local/global routing is structural. Retry caps are engineering safeguards,
  not evidence that a failure is non-local.
- Blocking boundaries fail closed when repair is unsuccessful.
- Hidden PGIR does not expose explicit taint labels in repair prompts.
- Interface-preserving local graph rewrites may add, remove, merge, or split
  nodes inside a repair region while preserving outside-node semantics and
  dependencies.
- Gold plans and official scorers are used only after execution for evaluation.

## Verification

Run:

```powershell
python smoke_frontier_structural.py
```

The structural smoke covers dependency inference, fan-in and fan-out repair,
joint frontiers, final-commit repair, fail-closed boundaries, hidden-taint
prompt behavior, global escalation, and interface-preserving local graph
rewrites.

## Remaining Caveats

- Repair proposal quality and semantic sufficiency verification remain active
  research problems.
- Benchmark datasets and external execution environments are not included in
  this repository.
