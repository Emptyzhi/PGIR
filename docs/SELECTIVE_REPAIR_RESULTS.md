# Selective Repair Diagnostic Results

## Scope

These diagnostics test why unrestricted full-trace retry can outperform PGIR
and whether PGIR should keep a cascading semantic verifier.

Method commits:

- `a86af38`: equal repair-time tool context, hard-contract-only conditions, and
  selective fallback;
- `cc98461`: full-trace acceptance semantics after fallback;
- `4f1a9cc`: identical full-trace fallback prompt and advisory soft semantic
  failures.

All runs use 10 fixed DisasterBench tasks with one seed. They are diagnostics,
not publishable main results.

## Five-Condition Diagnostic

Artifact:
`G:\develop\AutoResearchClaw\artifacts\pgir-selective-repair-diagnostic-10`

| Condition | Primary | Mean tokens | Mean re-executed | Stopped |
|---|---:|---:|---:|---:|
| Full-trace retry | 0.6000 | 4799.1 | 1.50 | 0 |
| Existing PGIR cascade | 0.6333 | 3482.7 | 0.00 | 1 |
| Selective fallback, first version | 0.6333 | 3694.2 | 0.00 | 1 |
| Hard-contract only | 0.3000 | 3041.3 | 0.00 | 0 |
| Hard-contract only plus fallback | 0.3000 | 3041.3 | 0.00 | 0 |

Hard-contract-only never activated because the DisasterBench adapter validates
plans against schemas but rarely produces real execution-time failures.
Removing semantic verification therefore reduces the method to the initial
plan and loses substantial score.

## Final Prompt-Parity Probe

Artifact:
`G:\develop\AutoResearchClaw\artifacts\pgir-selective-fallback-parity-10`

| Condition | Primary | Mean tokens | Mean re-executed | Stopped |
|---|---:|---:|---:|---:|
| Full-trace retry | 0.5333 | 5028.2 | 1.90 | 0 |
| Selective verified fallback | 0.5667 | 3970.7 | 0.80 | 0 |

Task-level comparison: PGIR has 1 win, 9 ties, and 0 losses. Relative to
full-trace retry in this probe, PGIR uses about 21% fewer approximate tokens and
58% fewer re-executed steps.

The selective condition recorded one full-trace fallback and two unresolved
soft semantic advisories. Neither advisory stopped execution.

## Structural Fault Injection

`fault_injection_diagnostic.py` verifies the locality assumption independently
of the DisasterBench scorer:

| Injected failure | Localizable | Local replay | Full replay | Saved steps |
|---|---:|---:|---:|---:|
| Late chain error | yes | 2 | 4 | 2 |
| Independent branch error | yes | 2 | 4 | 2 |
| Multi-root fan-in error | yes | 3 | 4 | 1 |
| Root plan error | no | 4 | 4 | 0 |

This confirms the structural claim: local repair has a cost advantage only
when the responsible frontier induces a strict subgraph. Root-level or broadly
propagated failures should use full replanning.

## Interpretation

1. Removing semantic verification entirely is not viable.
2. Soft semantic verification should propose repair but must not stop execution.
3. Once local repair cannot be validated, fallback should use the same prompt
   and acceptance semantics as full-trace retry.
4. The final parity probe supports the selective policy, but the sample is too
   small and full-trace scores vary substantially across repeated 10-task runs.
5. DisasterBench mostly measures plan correction. Many accepted PGIR changes
   are pre-execution structural pruning with zero re-executed steps, so it is
   insufficient evidence for runtime ancestor-path repair.

## Next Experimental Gate

Do not proceed directly to a full DisasterBench main experiment. First evaluate
the selective policy on a benchmark or controlled harness with real
execution-time faults and observable side effects. Report results separately
for:

- strict-local single-root failures;
- multi-root but still localizable failures;
- non-local/root-plan failures;
- verifier false positives.

The key paper claim should be conditional rather than universal: PGIR preserves
correct prior work when a causally responsible repair frontier is both
localizable and externally validated; otherwise it falls back to full-trace
replanning.
