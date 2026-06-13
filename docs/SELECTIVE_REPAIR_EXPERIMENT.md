# Selective Repair Diagnostic

This experiment isolates three possible reasons that full-trace retry can
outperform PGIR:

1. unequal repair-time tool context;
2. false-positive soft semantic verification;
3. fail-closed behavior after a rejected local repair.

All PGIR repair conditions receive the same complete public tool catalog as
full-trace retry. The diagnostic phase compares:

- `full_trace_retry_control`: unrestricted complete-plan regeneration;
- `pgir_hidden_taint_ancestor_repair`: existing hard plus soft semantic cascade;
- `pgir_selective_verified_fallback`: existing cascade with one full replan
  after a local candidate cannot be validated. Once triggered, fallback uses
  full-trace acceptance semantics: a parsed complete plan is accepted without
  another veto from the verifier that requested fallback. It also uses the
  same unrestricted repair prompt as the full-trace baseline. Unresolved soft
  semantic warnings are advisory and cannot stop execution;
- `pgir_hard_contract_only`: only tool execution, required input, declared
  output, and schema mismatches can block execution;
- `pgir_hard_contract_selective_fallback`: hard-contract repair plus one full
  replan fallback.

The fallback is not treated as a PGIR success. It is recorded separately through
`selective_fallbacks`, `fallback_reason`, and `fallback_global_source`. The
experiment asks whether PGIR loses because localization is wrong or because an
imperfect verifier prevents an otherwise recoverable trajectory from finishing.

Run:

```powershell
cd disasterbench
python main.py --phase selective_repair_diagnostic_10
python analyze_selective_repair.py <results-dir>/results.json
```
