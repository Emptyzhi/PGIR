# Fixed 10-task baseline fairness results

Artifact:
`G:\develop\AutoResearchClaw\artifacts\pgir-baseline-fairness-10\results.json`

The run completed 100/100 units with no errors. The baseline fairness audit,
tuning-parity audit, and prompt-contamination audit all passed. All conditions
received the same initial plan for each task, model, and seed.

## Main results

| Condition | Primary | Approx. tokens | LLM calls | Search calls | Re-executed steps |
| --- | ---: | ---: | ---: | ---: | ---: |
| PGIR selective verified fallback | **0.6333** | 3835.4 | 1.50 | 0.00 | 0.20 |
| Full-trace retry | 0.5333 | 4950.8 | 2.00 | 0.00 | 1.60 |
| Reflexion verbal retry | 0.5000 | 7350.4 | 2.70 | 0.00 | 2.00 |
| Post-tool reflection + RAG | 0.3333 | 5106.9 | 2.00 | 0.70 | 3.20 |
| No repair | 0.3000 | 3041.3 | 1.00 | 0.00 | 0.00 |
| Binary checkpoint rollback | 0.3000 | 4850.3 | 2.00 | 0.00 | 1.70 |
| Local leaf retry | 0.3000 | 6462.3 | 2.80 | 0.00 | 1.50 |
| No-contract retry | 0.3000 | 4916.6 | 2.00 | 0.00 | 3.40 |
| AgentFixer single-trace online adaptation | 0.3000 | 4957.9 | 2.00 | 0.00 | 3.30 |
| AgentRx failure-localization adaptation | 0.2000 | 6176.0 | 2.60 | 0.00 | 3.70 |

PGIR versus the strongest alternatives:

- Full-trace: 2 wins, 8 ties, 0 losses.
- Reflexion: 3 wins, 7 ties, 0 losses.
- Post-tool reflection + RAG: 4 wins, 6 ties, 0 losses.
- AgentFixer adaptation: 5 wins, 5 ties, 0 losses.
- AgentRx adaptation: 6 wins, 4 ties, 0 losses.

## Scorer decomposition

| Condition | Overall | Tools | Params | Dependencies |
| --- | ---: | ---: | ---: | ---: |
| PGIR selective verified fallback | **0.6333** | **0.8000** | **0.5000** | **0.6000** |
| Full-trace retry | 0.5333 | 0.6000 | 0.5000 | 0.5000 |
| Reflexion verbal retry | 0.5000 | 0.5000 | 0.5000 | 0.5000 |
| Post-tool reflection + RAG | 0.3333 | 0.4000 | 0.3000 | 0.3000 |
| AgentFixer adaptation | 0.3000 | 0.3000 | 0.3000 | 0.3000 |
| AgentRx adaptation | 0.2000 | 0.2000 | 0.2000 | 0.2000 |

## Interpretation

PGIR's advantage on this subset comes mainly from preserving or selecting the
smallest sufficient tool set. It improves two tasks over full-trace without
losing any task, while using fewer approximate tokens and re-executing fewer
steps.

Full-trace and Reflexion are strong because they can rewrite the whole plan, but
they also spend more calls and tokens. AgentFixer-style diagnosis frequently
produced plausible root-cause prose yet expanded the plan to four to six tools,
so its final score remained equal to no repair. AgentRx's first-failure suffix
regeneration was even more prone to unnecessary suffix expansion. This supports
the distinction between diagnosing a failure and selecting a minimal effective
repair target.

## Limitations

- This is a diagnostic subset of ten DisasterBench tasks, one model, one seed.
  It is not a paper-level effectiveness claim.
- Reflexion, AgentRx, post-tool reflection + RAG, and AgentFixer are
  paper-inspired harness adaptations, not official repository reproductions.
- AgentFixer is originally an offline diagnosis and remediation framework; its
  online retry condition is explicitly an adaptation.
- DART is not included. It evaluates semantic rollback recoverability and does
  not expose a directly comparable per-task DisasterBench repair interface.
- DisasterBench scoring measures structured plan agreement. A runtime execution
  benchmark is still required to validate PGIR's contamination-frontier and
  localized re-execution claims.
