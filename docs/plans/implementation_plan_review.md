# Implementation Plan — Review Comments

**Document**: `implementation_plan.md`
**Verdict**: Solid engineering plan. A few things to fix before you start coding.

---

## 🔴 Critical — Fix Before Starting

### 1. Corrupted Mermaid diagram (line ~251)

The Architecture Overview diagram contains a stray `</thinking>` tag immediately before the closing code fence. This will break rendering and is clearly a copy-paste artifact.

```
    EvaluationPipeline --> Z3Validator

</thinking>                          ← DELETE THIS LINE

```
```

Remove that line. Nothing else in the diagram needs changing — the graph itself is correct.

---

### 2. Reward functions are built in the wrong phase

This is the most substantive structural issue in the plan. The todo list and Phase 5 both place `CompositeRewardFunction` and its 5 components during "Synthetic Scaling" (Weeks 6–10). But rewards are only consumed by GRPO, which doesn't run until Phase 6 (Weeks 11–14, and marked Optional).

You are building GRPO infrastructure during a phase whose stated goal is scaling synthetic data. These are unrelated workstreams.

**Move reward function TDD (Cycle 9) entirely into Phase 6.** Phase 5 becomes what it says it is: synthetic generation, quality control, Z3 integration, Stage 1–2 training, and HF publishing. If GRPO turns out to be unnecessary after Stage 2 evaluation, you've saved yourself the effort of building reward infrastructure you never use.

Concretely, in the todo list, `reward-tests` and `reward-impl` should sit after `z3-integration`, not between `evaluation-impl` and `base-trainer`.

---

### 3. ContentQualityValidator is referenced but never defined

It appears in three places — the test directory structure (`test_content_validator.py`), the TDD Scope table under "Domain", and the Test-First Components list — but there is no definition anywhere: no design pattern sketch, no todo item, no description of what it actually checks.

This matters because the boundary between ContentQualityValidator and the Tier 2 LLM Judge is unclear. Tier 2 scores Nyaya methodology quality via GPT-4. If ContentQualityValidator does something different, say it. If it doesn't, remove it from the plan to avoid building a redundant component.

Likely candidates for what it's meant to do:
- Validate that Pratyaksha contains only observable facts (no inferred content)
- Check that Udaharana contains the "Wherever X, there is Y" universal rule structure
- Verify Tarka isn't tautological

If that's the intent, add a design pattern sketch alongside the others and a corresponding todo item. Otherwise, cut it.

---

## 🟡 Structural — Worth Reorganising

### 4. The todo list order doesn't match the phase order

A reader following the todos sequentially would build reward functions before trainers, and trainers before the evaluation pipeline. The actual execution order (per the Implementation Phases section) is:

```
Domain Models → Validator → Parser → Converter → Config →
Trainer → Checkpoint → Evaluation → Seed Examples →
Synthetic + Z3 → GRPO + Rewards → HF Publishing
```

The todo list should follow this sequence. Right now it reads like it was assembled as items were thought of rather than as a build order. For a TDD-driven plan where order is the whole point, this is worth fixing.

---

### 5. Seed example creation is undersold

`seed-examples` — "Create 5 gold-standard Stage 0 seed examples" — is listed as a single todo item at the same level as "create CLI commands." Per spec.md, this is 15 hours of highly skilled work that leverages unique Sanskrit/Nyaya expertise. It is the single most important deliverable in Stage 0, and arguably in the entire project. Everything downstream (training, evaluation, synthetic generation) depends on these examples being correct.

Two suggestions:

First, break it into 5 explicit todos, one per problem, matching the progression defined in spec.md:
```
seed-example-01: 3-variable direct constraints (Alice/Bob/Carol)
seed-example-02: 3-variable relational constraints (race positions)
seed-example-03: 4-variable mixed constraints (houses)
seed-example-04: 4-variable negative-only constraints (languages)
seed-example-05: 5×5 Zebra puzzle (integration test)
```

Second, these should appear in Phase 2 (alongside the parser and converter), not in Phase 5. You need at least one complete example to write meaningful parser and validator tests. The `valid_example.md` fixture *is* a seed example. Writing it as a "test fixture" and separately as "content work" in a later phase is an artificial split — create problem 1 first, use it as your fixture, and the parser tests become grounded in a real Nyaya solution rather than a stub.

---

### 6. Phase 5 does too much

Phase 5 spans Weeks 6–10 (four weeks) and currently contains: reward functions (moved to Phase 6 per comment #2), synthetic generation prompts, the quality control pipeline, Z3 integration, Stage 1 and Stage 2 training runs, *and* HuggingFace publishing. That's five distinct workstreams.

After moving rewards out, it's still heavy. Consider splitting:

```
Phase 5a (Wk 6-7): Synthetic generation + quality control pipeline
Phase 5b (Wk 8-9): Z3 integration + Stage 1-2 training execution
Phase 5c (Wk 10):  Evaluation, decision gate, HF publish if GO
```

This gives you natural checkpoints. If synthetic generation quality is poor after Phase 5a, you refine prompts before burning compute on training.

---

### 7. Tier 3 handler has no todo or TDD cycle

Tier1Handler and Tier2Handler each have explicit TDD cycles (Cycles 6 and 7). Tier3Handler (`ManualReviewHandler`) appears in the file structure and the architecture diagram but has no corresponding todo item, no TDD cycle, and no implementation phase assignment.

This is probably intentional — manual review is inherently interactive, not something you unit-test in the traditional sense. But you still need the *interface*: how does a human reviewer receive examples, record decisions, and feed results back into the pipeline? A CLI command, a simple web UI, a structured file format for recording verdicts?

At minimum, add a todo for the Tier 3 interface — even if it's just a CLI script that reads from a queue directory and writes accept/reject decisions — and note it as integration-tested rather than TDD.

---

## 🟢 Observations — Good Calls Worth Noting

**TDD scope calibration is right.** Strict TDD on domain and application layers, integration-only on ML infrastructure wrappers. You cannot meaningfully write a failing test for "does Unsloth load this model correctly" before calling Unsloth. The coverage omit on `infrastructure/ml/*` is pragmatic and justified.

**The fixture set is well-chosen.** `valid_example.md`, `missing_tarka.md`, `invalid_udaharana.md`, `malformed_yaml.md` — four fixtures that exercise the four failure modes the validator needs to catch. Clean.

**Commit pattern is disciplined.** Two commits per TDD cycle (test first, then implementation) gives you a clean, auditable history where you can see exactly what was tested before what was built. Good practice, especially for a project that will be open-sourced.

**pytest marker strategy is sound.** `slow`, `gpu`, `integration`, `e2e` — lets you run fast unit tests during active TDD without waiting for GPU warmup or network calls. The `uv run pytest -m "not slow and not gpu"` command will be your workhorse.

**uv + frozen lockfile in Docker is the right call.** `uv sync --frozen` means the container build is deterministic. Anyone pulling your repo and running `docker compose up` gets exactly your dependency graph. Critical for reproducibility.

**The production Docker stage should probably go.** This is a research project. Its output is a model and dataset on HuggingFace, not a running service. The production stage adds a layer of complexity (ENTRYPOINT, stripped image) that has no consumer yet. Keep the development stage, drop production until/unless Stage 4 actually materialises.

---

## Summary

| # | Severity | Issue | Action |
|---|----------|-------|--------|
| 1 | 🔴 | `</thinking>` corrupts Mermaid diagram | Delete the line |
| 2 | 🔴 | Reward functions built in wrong phase | Move to Phase 6 |
| 3 | 🔴 | ContentQualityValidator undefined | Define it or cut it |
| 4 | 🟡 | Todo list order ≠ execution order | Reorder to match phases |
| 5 | 🟡 | Seed examples underweighted | Break into 5 todos, move earlier |
| 6 | 🟡 | Phase 5 overloaded | Split into 5a/5b/5c |
| 7 | 🟡 | Tier 3 handler has no todo | Add interface todo |
