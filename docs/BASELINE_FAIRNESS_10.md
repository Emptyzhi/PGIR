# Fixed 10-task baseline fairness experiment

The `baseline_fairness_10` phase compares PGIR with all runnable comparison
baselines on the same ten DisasterBench tasks.

## Fairness policy

- Same fixed task IDs, model, seed, temperature, initial-plan token cap, and
  repair token cap.
- Every condition receives an identical cached initial plan. Results include a
  SHA-256 `initial_plan_fingerprint`, and the harness fails its fairness audit
  when fingerprints or condition coverage differ.
- Every repair method sees the complete public task tool catalog. Gold plans
  and scorer outputs are forbidden from prompts.
- Method-intrinsic resources are retained and reported rather than hidden:
  Reflexion may use an extra reflection call, and post-tool RAG may use Tavily.
  Results report approximate tokens, LLM calls, repair LLM calls, search calls,
  and re-executed steps.

## Baseline fidelity

`reflexion_verbal_retry`, `agentrx_diagnosis_failure_localization`, and
`post_tool_reflection_rag_repair` are paper-inspired harness implementations,
not executions of official repositories.

AgentFixer is an offline diagnosis and remediation-recommendation framework,
not a per-task online retry agent. The
`agentfixer_single_trace_recommendation_retry` condition is therefore explicitly
an online adaptation of its hybrid validation, single-trace root-cause
analysis, and remediation recommendation path. It must not be reported as an
official AgentFixer reproduction.

Run:

```powershell
$env:DISASTERBENCH_ROOT='G:\develop\AutoResearchClaw\artifacts\benchmarks\DisasterBench_Open'
$env:PGIR_RESULTS_DIR='G:\develop\AutoResearchClaw\artifacts\pgir-baseline-fairness-10'
python disasterbench\main.py --phase baseline_fairness_10
python disasterbench\analyze_baseline_fairness.py `
  G:\develop\AutoResearchClaw\artifacts\pgir-baseline-fairness-10\results.json
```
