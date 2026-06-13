# PGIR New Positioning: Failure Propagation Repair

## One-Sentence Thesis

PGIR reframes agent repair as **failure propagation repair**: instead of repairing the observed failure site, it traces causal contamination through execution provenance and repairs the responsible repair frontier at runtime.

## Core Claim

Existing agent repair methods often assume that the observed failure location is the correct repair target. This assumption breaks in long-horizon tool-use agents because failures are often the end result of causal error propagation.

The central distinction is:

- Observed failure site is where the failure becomes visible.
- Responsible repair target is where the execution became semantically contaminated.

PGIR's contribution is not simply repairing earlier than post-hoc methods. Runtime intervention is the mechanism. The deeper contribution is choosing the correct repair object under failure propagation.

## Relation to DART

DART's core claim can be summarized as:

> Not every checkpoint is recoverable.

PGIR's analogous claim is:

> Not every failure site is the right repair target.

DART studies semantic recoverability for checkpoint rollback: whether a controller-legal checkpoint is semantically admissible under downstream commitments and effect boundaries.

PGIR studies causal repair target selection for tool-using agents: whether the node where failure is observed is actually the node or subgraph that should be repaired.

Therefore, checkpoint rollback methods should be discussed as adjacent related work and included as recovery controls, but PGIR should not be framed as a checkpoint rollback method.

## Problem Definition

### Failure Propagation Repair

Given a tool-agent execution graph with actions, dependencies, intermediate artifacts, and aggregation boundaries, a failure may be observed at node `v`, but its responsible cause may lie in one or more upstream nodes.

PGIR defines repair as selecting and rewriting the minimal responsible frontier whose correction can eliminate currently observed execution-critical violations while preserving unaffected work.

### Key Concepts

**Observed Failure Site**

The node or boundary where a violation becomes externally visible.

**Execution-Critical Violation**

A discrete contract violation that makes downstream execution semantics untrustworthy. This is not a continuous risk score.

Examples:

- missing required input
- wrong output schema
- broken dependency
- invalid parameter provenance
- stale or mismatched artifact
- aggregation contradiction
- consumed output from invalidated ancestor
- irreversible side-effect boundary conflict

**Non-Blocking Deviation**

A quality issue that does not invalidate downstream execution semantics.

Examples:

- retrieval result is weaker but structurally valid
- summary is less detailed
- alternative valid tool choice
- minor formatting issue

**Responsible Repair Frontier**

The minimal set of ancestor nodes or subgraphs whose joint correction can eliminate all currently observed blocking violations.

This generalizes responsible ancestor path:

- If one causal chain is responsible, the frontier is an ancestor path.
- If multiple sibling branches jointly break an aggregation, the frontier contains multiple roots.
- If the frontier approaches the whole graph, PGIR should escalate to global replan rather than repeatedly perform local repair.

## Runtime Mechanism

PGIR should not implement eager micro-repair:

```text
violation -> repair -> violation -> repair -> ...
```

This can be more expensive than post-hoc repair when many nodes are wrong or when aggregation exposes multiple independent errors.

Instead, PGIR should implement:

```text
Detect continuously, repair selectively.
```

### Execution Loop

```text
for each executable node:
    verify input contracts and consumed provenance
    if non-blocking deviation:
        log deviation and continue
    if execution-critical violation:
        mark contamination evidence
        defer repair unless boundary policy requires immediate repair

    execute node

    verify output contract
    classify violation
    update contamination graph

    if node is repair boundary:
        select responsible repair frontier
        if frontier is local:
            perform joint localized repair
            invalidate affected descendants
            continue from repaired frontier
        else:
            escalate to broader subgraph repair or global replan
```

### Repair Boundaries

PGIR should plan repair at important boundaries rather than repairing every violation immediately:

- aggregation or join boundary
- before irreversible side effects
- before final answer
- before tainted output is consumed by multiple downstream nodes
- when accumulated contamination evidence exceeds local repair assumptions

## Why Not Taint Scores

PGIR should avoid `taint_score > threshold` as a core mechanism.

Reasons:

- Different node types have incomparable error spaces.
- Thresholds invite arbitrary sensitivity questions.
- The paper would become a tuning paper rather than a repair-target paper.

Instead, PGIR should use discrete violation classification:

```json
{
  "status": "pass | non_blocking_deviation | blocking_taint",
  "violation_type": "missing_required_input | broken_dependency | aggregation_inconsistency | ...",
  "responsible_nodes": [2, 4],
  "affected_nodes": [5, 6]
}
```

This mirrors DART's structural style:

- DART asks whether rollback is admissible.
- PGIR asks whether a violation is execution-critical and where its responsible repair frontier lies.

## Escalation Policy

Localized repair is only valid when most of the execution graph remains trustworthy.

If the contaminated frontier is too large, repeated local repair is the wrong strategy. PGIR should be fail-closed and escalate.

Suggested policy:

```text
1. joint local frontier repair
2. broader ancestor subgraph repair
3. full trace replan
4. fail-safe abort or human handoff
```

Each repair frontier should have a bounded retry budget. PGIR should never loop indefinitely on the same blocking contamination.

## Contributions

### Contribution 1: Failure Propagation View of Agent Repair

PGIR identifies a hidden assumption in existing agent repair methods: that the observed failure location is the correct repair target. It shows that long-horizon tool-use failures often arise from causal contamination that propagates through dependencies and aggregations.

### Contribution 2: Contract-Provenance Contamination Graph

PGIR uses contract verification and provenance/taint tracking to model failures as execution-critical contamination events over an execution graph, rather than isolated failed steps.

### Contribution 3: Responsible Repair Frontier

PGIR introduces repair frontier selection: jointly repairing the minimal responsible ancestor subgraph needed to eliminate currently observed blocking violations, while preserving unaffected sibling work.

### Contribution 4: Deferred Runtime Intervention

PGIR continuously monitors execution but repairs selectively at boundary points. This avoids expensive eager micro-repair while still intervening before contamination fully propagates to irreversible effects or final output.

## Baseline Story

PGIR should not claim its core contribution is checkpoint rollback.

Recommended baseline structure:

| Category | Baselines | Purpose |
| --- | --- | --- |
| Post-hoc self-repair | Reflexion, ReAct retry, self-reflection repair | Show that repairing after final failure is late and coarse |
| Failure diagnosis | AgentRx, AgentFixer-style diagnosis | Show that locating observed failures is not enough |
| Recovery controls | full-trace retry, whole-task rerun, local leaf retry, checkpoint rollback | Show that replay or mechanical rollback does not solve repair target selection |
| PGIR ablations | no provenance, no contract, leaf-only, eager micro-repair, full-trace PGIR | Show why each PGIR component matters |

## Essential Experiments

### 1. Failure Site vs Root Cause

Measure whether the observed failure site differs from the responsible repair frontier.

Metrics:

- failure-site localization accuracy
- responsible-frontier localization accuracy
- ancestor/frontier precision and recall

### 2. Leaf Repair vs Frontier Repair

Compare:

- local leaf retry
- responsible ancestor path repair
- responsible repair frontier repair

Expected result:

Leaf repair fails when the observed failure is downstream of the true cause.

### 3. Eager Micro-Repair vs Deferred Frontier Repair

Compare:

- repair immediately on every violation
- accumulate contamination evidence and repair at boundaries

Expected result:

Deferred frontier repair reduces repeated repair cycles while preserving repair success.

### 4. Local Repair vs Global Replan

Show that PGIR escalates when contamination is no longer local.

Metrics:

- frontier size / graph size
- replayed actions
- repair calls
- successful recovery rate

### 5. Post-Hoc Repair Comparison

Compare PGIR against post-hoc repair methods under long-horizon tool-use tasks.

Important metrics:

- recovery success
- replay frontier size
- unnecessary sibling disturbance
- number of LLM repair calls
- contamination depth before repair
- irreversible side-effect violations avoided

## Implementation Requirements

The current PGIR implementation must not be merely:

```text
run full plan -> diagnose -> repair
```

To support this paper story, the implementation must include:

- runtime action-boundary verification
- dependency-consumption verification
- aggregation-boundary verification
- discrete violation classification
- contamination evidence graph
- repair frontier selection
- deferred boundary repair
- bounded retry and escalation
- separate metrics for repair calls, replayed actions, frontier size, and sibling preservation

## Paper Positioning

The paper should not be titled or framed as a generic agent repair method.

Better framing:

> Failure Propagation Repair for Tool-Using Agents

Alternative phrasing:

> Causally Localized Runtime Repair for Tool-Using Agents

Central slogan:

> Existing repair methods fix where failure is observed. PGIR fixes where failure propagation originates.

