# Stage 1 Plan Review (Rigor Check)

**Date**: 2026-02-02  
**Scope**: Stage 1 execution vs plan in `CLAUDE.md`  
**Goal**: Verify which plan items were completed, partial, or missing.

---

## 1) Plan Requirements (from CLAUDE.md)

Stage 1: Minimum Viable Reasoner
- Create **50 gold-standard examples** (constraint + Boolean SAT)
- Fine-tune **DeepSeek-R1-Distill-Llama-8B** with strong LoRA
- Success criteria:
  - **>90% format adherence**
  - **60–70% accuracy** on held-out problems

---

## 2) Evidence Collected (Artifacts)

**Training + data**:
- `data/seed_examples/stage_zero/` (20 examples)
- `data/seed_examples/stage_one/` (35 examples)
- `data/training/stage_1.jsonl` (55 examples)
- `scripts/prepare_stage1_training_data.py`
- `scripts/train_stage1.py`
- `models/stage_1/` (final adapter + checkpoints)

**Evaluation**:
- `results/stage_1_evaluation.json` (10 examples)
- Evaluation run via `scripts/evaluate_stage0.py` with Stage 1 model

**Publishing + deployment**:
- HF adapter: `qbz506/nyaya-deepseek-8b-stage1`
- HF full model: `qbz506/nyaya-deepseek-8b-stage1-full`
- HF dataset: `qbz506/pramana-nyaya-stage1`
- Space: `qbz506/pramana-nyaya-demo`

**Plots and numeric appendix**:
- `docs/stage_1_paper_appendix.md`
- `docs/figures_stage1_v2/stage1_train_loss.csv`
- `docs/figures_stage1_v2/stage1_eval_loss.csv`
- `docs/figures_stage1_v2/stage1_train_loss.png`
- `docs/figures_stage1_v2/stage1_eval_loss.png`

---

## 3) Plan vs Execution (Checklist)

**A) Data requirements**
- **Planned**: 50 gold-standard examples
- **Executed**: 55 total examples (20 Stage 0 + 35 Stage 1)
- **Status**: **Partial**
  - Quantity target met (55)
  - Provenance unclear: not all 55 are explicitly validated as “gold-standard”

**B) Model + training**
- **Planned**: DeepSeek-R1-Distill-Llama-8B + strong LoRA
- **Executed**: `unsloth/DeepSeek-R1-Distill-Llama-8B-bnb-4bit` with LoRA rank 64
- **Status**: **Complete**

**C) Success criteria: format adherence**
- **Planned**: >90%
- **Observed**: 40% (4/10)
- **Status**: **Failed**

**D) Success criteria: accuracy**
- **Planned**: 60–70% on held-out
- **Observed**: 100% semantic correctness on evaluation set
- **Status**: **Exceeded**

**E) Evaluation rigor**
- **Planned**: Format adherence + correctness (and later Z3 in Stage 2)
- **Executed**: Tier 1 structural validation + semantic similarity
- **Missing**: Tier 2 LLM judge (no API key provided)
- **Status**: **Partial**

**F) Full model artifacts**
- **Planned**: Merged full model
- **Executed**: `qbz506/nyaya-deepseek-8b-stage1-full`
- **Status**: **Complete**

**G) Deployment**
- **Planned**: Space demo
- **Executed**: Stage 0 + Stage 1 in `qbz506/pramana-nyaya-demo`
- **Status**: **Complete**

---

## 4) Gaps and Incompleteness

1) **Format adherence is far below target**  
   - This is the primary unmet success criterion.

2) **Tier 2 LLM judge evaluation is missing**  
   - Not implemented due to missing API credentials.

3) **Gold-standard validation unclear**  
   - We combined Stage 0 + Stage 1 seeds into 55 examples, but a gold-standard verification checklist is not yet documented.

4) **Z3 verification not applied**  
   - Tier 3 Z3 is present but skipped for all Stage 1 eval examples (none flagged as Z3 verifiable).

---

## 5) Recommendations (Next Actions)

1) **Stage 2: tighten format enforcement**  
   - Add parser-based filtering and regenerate any output that fails strict headers.

2) **Add Tier 2 LLM judge**  
   - Once API key is available, run `scripts/run_tier2_judge.py` over Stage 1 outputs.

3) **Gold-standard checklist**  
   - Define acceptance criteria per example and annotate dataset accordingly.

4) **Expand evaluation set**  
   - 10 examples is too small for stable confidence intervals; target 50–100.

---

## 6) Final Determination

Stage 1 **successfully delivered** training, artifacts, and deployment, but **does not meet the formal success criteria** due to low format adherence and incomplete Tier 2 judging. The stage is **functionally usable** for reasoning demos but **not yet a validated Nyaya-structured reasoner** per plan.
