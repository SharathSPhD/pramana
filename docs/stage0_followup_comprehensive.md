# Stage 0 Follow-Up Action Report (Updated)

**Date**: 2026-01-31  
**Context**: Post-corrective iteration analysis  
**Status**: ✅ **Stage 0 COMPLETE** - Format adherence verified at 100%

---

## Executive Summary

Stage 0 has been successfully completed after a corrective iteration. The model now demonstrates **100% format adherence** (2/2 held-out examples) on the 6-phase Nyaya structure, exceeding the ≥80% success criterion. However, several critical observations and next steps emerge from this completion that must be addressed before Stage 1.

**Current State**:
- ✅ Format learning proven: Model can learn and reproduce 6-phase structure
- ✅ Deployed: HuggingFace Hub, Ollama/OpenWebUI, demo Space
- ⚠️ Answer accuracy: 0/2 exact match (strict string matching issue)
- ⚠️ Semantic correctness: Appears correct but uses different phrasing
- ❌ Hyperparameters still below spec (r=32 vs 64-128)
- ❌ No quantitative content quality metrics logged

**Decision**: **CONDITIONAL GO for Stage 1** with mandatory fixes listed below.

---

## What Changed: Initial Failure → Corrective Success

### Initial Run (Failed)
**Metrics**: 0% format adherence, 0% accuracy  
**Issue**: Model completely ignored Nyaya structure  
**Root Cause**: According to implementation report:
- Undertrained (insufficient epochs or capacity)
- Hyperparameters too conservative
- No held-out validation during training

### Corrective Run (Succeeded)
**Metrics**: 100% format adherence (2/2), semantically correct answers  
**Changes Made**:
- Training data refined
- Held-out test set created (2 examples)
- Evaluation pipeline implemented (`scripts/evaluate_stage0.py`)
- Model deployed to HF Hub + Ollama

**Evidence**:
```
results/stage_0_corrected_evaluation_v7.json:
- Format adherence: 100% (2/2 test examples with all 6 phases)
- Phase completeness: 6/6 phases present
- Answer correctness: 0/2 exact match (but semantically equivalent)
```

---

## Critical Analysis: What Stage 0 Proves and Doesn't Prove

### ✅ What We Proved

**1. Format Learnability**
The fundamental hypothesis is validated: LLMs can learn the 6-phase Nyaya structure through fine-tuning. With only 3-5 training examples, the model:
- Generates all 6 phases in correct order
- Populates each phase with appropriate content
- Maintains structure on unseen problems

This is **non-trivial** — most structured output approaches require extensive prompting or constrained decoding. Fine-tuning embeds the structure.

**2. Minimal Data Sufficiency**
3-5 examples are enough to teach format adherence. This validates the "quality over quantity" approach and suggests Stage 1 (50 examples) should provide robust performance.

**3. Infrastructure Viability**
The full pipeline works: markdown parsing → training → evaluation → deployment. DGX Spark + Unsloth + LoRA is functional.

### ⚠️ What We Haven't Proven

**1. Reasoning Quality Within Format**
100% format adherence tells us the model learned *what phases to generate*, not whether the *content of those phases* is sound Nyaya reasoning.

**Critical distinction**:
- ✅ Model generates "Samshaya", "Pramana", "Pancha Avayava", etc.
- ❓ Does Pratyaksha contain only observable facts (not inferred)?
- ❓ Do Udaharana provide universal rules ("Wherever X, there is Y")?
- ❓ Does Tarka actually test conclusions via reductio ad absurdum?

**Gap**: No automated content quality metrics. The evaluation script checks structure but not semantic correctness of Nyaya methodology.

**2. Generalization Beyond 2 Test Examples**
2/2 is statistically insignificant. Could be lucky. Need ≥10 held-out examples to establish reliability.

**3. Model Capacity Under Real Constraints**
Current hyperparameters (r=32, max_seq_length=2048) are below spec recommendations:
- Spec: r=64-128, max_seq_length=4096+
- Actual: r=32, max_seq_length=2048

The model succeeded *despite* under-capacity, which suggests either:
- The spec overestimated requirements (good news)
- The test problems are too easy (concerning)
- We got lucky with these 2 examples (needs more testing)

**4. Answer Correctness**
0/2 exact match accuracy is alarming even if semantically correct. Two explanations:
- **Optimistic**: Model reasoning is correct, just different phrasing
- **Pessimistic**: Model generates plausible-sounding but wrong answers in valid format

**Gap**: Need semantic similarity scoring, not just exact string match.

---

## Critical Findings from Implementation Reports

From analyzing `stage_0_comprehensive_report.md` and `implementation_report.md`:

### 🔴 Critical Issues Remaining

**1. No Content Quality Metrics**
The evaluation measures:
- ✅ Phase presence (binary: present/absent)
- ✅ Phase ordering (binary: correct/incorrect)
- ❌ Phase content quality (NOT measured)

**Example**: Does the Pratyaksha section cite only observable facts, or does it mix inferences? **We don't know.**

**2. Hyperparameter-Spec Mismatch Unresolved**
Despite corrective iteration, hyperparameters still don't match project guidance:

| Parameter | Spec | Actual | Status |
|-----------|------|--------|--------|
| LoRA rank | 64-128 | 32 | ❌ BELOW |
| Sequence length | 4096+ | 2048 | ❌ BELOW |
| Epochs | 10-15 | Unknown | ❓ |
| Batch size | 2-4 | 1 | ❌ BELOW |

**Why this matters**: Stage 1 (50 examples) will have higher complexity. Insufficient capacity now could mean failure at scale.

**3. Answer Correctness Evaluation Broken**
From comprehensive report:
> "⚠️ Answer correctness: 0/2 exact-match (but **semantically correct** — evaluation uses strict string matching)"

If answers are semantically correct but fail string matching, the evaluator is wrong, not the model. This must be fixed.

### 🟠 High Priority Issues

**4. No Training Metrics Logged**
From original critical review:
> "Only training loss is logged; no structural/format metrics or validation loss"

We can't tell:
- Did the model learn gradually or suddenly?
- Are there signs of overfitting?
- What's the optimal stopping point?

**5. Deployment Without Benchmarking**
Model deployed to HuggingFace and Ollama **without comparative evaluation**:
- No baseline: How does untuned model perform?
- No benchmark: Performance on LogicBench, ProntoQA, etc.?
- No ablation: Does Nyaya structure help or hurt vs. standard CoT?

**6. Statistical Insignificance**
2 test examples is too few. Confidence interval on 100% with n=2 is [15.8%, 100%]. Even 5/5 would be [47.8%, 100%]. Need ≥10 examples for meaningful statistics.

### 🟡 Medium Priority Issues

**7. Z3 Verification Not Exercised**
From comprehensive report:
> "All seed examples marked `z3_verifiable: true`"

But no evidence Z3 verification actually ran. The `Z3Verifier` component exists but may not be integrated into evaluation pipeline.

**8. Tier 2 and Tier 3 Evaluation Unimplemented**
Only Tier 1 (structural validation) is working:
- Tier 2 (LLM judge for content quality): Not implemented
- Tier 3 (human review): Not implemented

**9. No Shortcut Detection Test**
From spec review, critical failure mode:
> "Model gets correct answer without using Nyaya structure"

**Test**: Run ablation with and without Nyaya instructions. If accuracy same → model found shortcut.

---

## Immediate Actions Before Stage 1

### Action 1: Fix Answer Correctness Evaluation ⏰ **2 hours**

**Issue**: 0/2 exact match when answers are semantically correct

**Root Cause**: String matching is too strict. Answers like "Bob has the dog" vs "Bob owns the dog" fail.

**Solution**: Implement semantic similarity scoring

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def semantic_match(predicted: str, ground_truth: str, threshold=0.85) -> bool:
    pred_emb = model.encode(predicted)
    gt_emb = model.encode(ground_truth)
    similarity = cosine_similarity(pred_emb, gt_emb)
    return similarity >= threshold

# Or: Use LLM-as-judge
def llm_judge_correctness(predicted, ground_truth, problem):
    prompt = f"""
    Problem: {problem}
    Ground truth answer: {ground_truth}
    Model answer: {predicted}
    
    Are these answers equivalent? Respond CORRECT or INCORRECT.
    """
    return gpt4(prompt).strip() == "CORRECT"
```

**Deliverable**: `scripts/evaluate_stage0.py` updated with semantic scoring

---

### Action 2: Implement Content Quality Validators ⏰ **8 hours**

**Goal**: Measure whether Nyaya phases contain appropriate content, not just correct labels

**Components to Build**:

**2a. Pratyaksha Validator** (2 hours)
```python
def validate_pratyaksha(pratyaksha_text: str, problem_text: str) -> float:
    """
    Score 0-1: How well does Pratyaksha contain only observables?
    
    Method:
    1. Extract claims from Pratyaksha section
    2. For each claim, check if it's verbatim or paraphrase from problem
    3. Score = (claims_from_problem) / (total_claims)
    """
    # Use NLP to extract claims
    # Compare to problem text via embedding similarity
    # Return score
```

**2b. Udaharana Universal Rule Checker** (2 hours)
```python
def validate_udaharana(udaharana_text: str) -> bool:
    """
    Check if Udaharana contains "Wherever X, there is Y" universal rule structure
    
    Method:
    1. Search for universal quantifiers: "Wherever", "In all cases", "Whenever"
    2. Check for general variables (not specific names)
    3. Verify consequence statement
    4. Check for "For example" transition to specific instance
    """
    patterns = [
        r"Wherever .+, there .+",
        r"In all cases where .+, .+",
        r"Whenever .+, .+"
    ]
    return any(re.search(p, udaharana_text) for p in patterns)
```

**2c. Tarka Meaningfulness Checker** (2 hours)
```python
def validate_tarka(tarka_text: str, nirnaya_answer: str) -> bool:
    """
    Check if Tarka actually tests the conclusion, not just restates it
    
    Method:
    1. Extract the negation being tested
    2. Verify it's the opposite of Nirnaya conclusion
    3. Check for derivation of contradiction
    4. Ensure conclusion isn't just "if X then X" tautology
    """
    # Detect tautologies
    # Verify contradiction structure
    # Check logical connection
```

**2d. Hetvabhasa Completeness** (1 hour)
```python
def validate_hetvabhasa(hetvabhasa_text: str) -> float:
    """
    Score 0-1: Proportion of 5 fallacy types explicitly checked
    
    Required fallacies:
    - Savyabhichara (erratic)
    - Viruddha (contradictory)
    - Prakaranasama (irrelevant)
    - Sadhyasama (unproved)
    - Kalaatita (mistimed)
    """
    fallacy_names = ["savyabhichara", "viruddha", "prakaranasama", 
                     "sadhyasama", "kalaatita"]
    found = sum(1 for f in fallacy_names if f.lower() in hetvabhasa_text.lower())
    return found / 5.0
```

**Integration** (1 hour):
```python
class ContentQualityValidator:
    def validate(self, nyaya_output: NyayaExample) -> ContentQualityResult:
        return ContentQualityResult(
            pratyaksha_score=validate_pratyaksha(nyaya_output.pratyaksha, problem),
            udaharana_valid=validate_udaharana(nyaya_output.udaharana),
            tarka_meaningful=validate_tarka(nyaya_output.tarka, nyaya_output.nirnaya),
            hetvabhasa_completeness=validate_hetvabhasa(nyaya_output.hetvabhasa),
            overall_score=weighted_average(...)
        )
```

**Deliverable**: `src/pramana/domain/validators/content_quality.py` + tests

---

### Action 3: Expand Held-Out Test Set ⏰ **4 hours**

**Goal**: Increase from 2 to 10 held-out examples for statistical significance

**Strategy**:
- Use pramana-003, pramana-005 (currently held-out) ✅
- Create 8 new examples specifically for testing (NOT training)

**New example distribution**:
| Type | Count | Purpose |
|------|-------|---------|
| Constraint satisfaction | 3 | Test core elimination method |
| Boolean SAT | 2 | Test logical equivalence |
| Transitive reasoning | 1 | Test chain building |
| Set membership | 2 | Test partitioning |
| Multi-step deduction | 2 | Test modus ponens chains |

**Quality criteria**:
- Each example verified with ground truth
- Comparable difficulty to training examples
- Diverse enough to test generalization

**Deliverable**: 
```
data/validation/stage_zero/
├── pramana-003.md
├── pramana-005.md
├── test-001-constraint.md
├── test-002-constraint.md
├── test-003-constraint.md
├── test-004-boolean.md
├── test-005-boolean.md
├── test-006-transitive.md
├── test-007-set.md
├── test-008-set.md
├── test-009-deduction.md
└── test-010-deduction.md
```

---

### Action 4: Run Comprehensive Validation with New Metrics ⏰ **2 hours**

**Goal**: Re-evaluate model with:
- 10 held-out examples (not 2)
- Semantic similarity for answers
- Content quality validators
- Statistical confidence intervals

**Script**:
```bash
python scripts/evaluate_stage0.py \
  --model_dir models/stage_0_corrected \
  --test_set data/validation/stage_zero/ \
  --samples_per_problem 5 \
  --output results/stage_0_final_validation.json \
  --enable_content_quality \
  --semantic_similarity
```

**Expected Output**:
```json
{
  "format_adherence": {
    "rate": 0.XX,
    "confidence_interval_95": [0.XX, 0.XX]
  },
  "answer_correctness": {
    "exact_match": 0.XX,
    "semantic_match": 0.XX,
    "confidence_interval_95": [0.XX, 0.XX]
  },
  "content_quality": {
    "pratyaksha_score": 0.XX,
    "udaharana_valid_rate": 0.XX,
    "tarka_meaningful_rate": 0.XX,
    "hetvabhasa_completeness": 0.XX,
    "overall_score": 0.XX
  },
  "per_example_results": [...]
}
```

**Deliverable**: `results/stage_0_final_validation.json` with confidence intervals

---

### Action 5: Execute Shortcut Detection Test ⏰ **3 hours**

**Goal**: Verify model genuinely uses Nyaya structure, not shortcuts

**Method**: Ablation study

```python
# Experiment 1: With Nyaya instructions (current)
prompt_nyaya = """
Solve this problem using the 6-phase Nyaya methodology:
Samshaya, Pramana, Pancha Avayava, Tarka, Hetvabhasa, Nirnaya.

Problem: {problem}
"""

# Experiment 2: Without Nyaya instructions (baseline)
prompt_baseline = """
Solve this problem step by step:

Problem: {problem}
"""

# Experiment 3: Nyaya instructions but removed from training
# (Test if base model already knows Nyaya from pre-training)
prompt_base_model = prompt_nyaya  # But use untuned model

results = {
    "nyaya_tuned_with_instructions": evaluate(tuned_model, prompt_nyaya, test_set),
    "nyaya_tuned_without_instructions": evaluate(tuned_model, prompt_baseline, test_set),
    "base_model_with_instructions": evaluate(base_model, prompt_nyaya, test_set)
}

# Analysis
if results["nyaya_tuned_without_instructions"].accuracy ≈ results["nyaya_tuned_with_instructions"].accuracy:
    print("WARNING: Model found shortcut, doesn't need Nyaya structure")
else:
    print("PASS: Nyaya structure contributes to reasoning")
```

**Deliverable**: `docs/stage0report/shortcut_detection.md` with ablation results

---

### Action 6: Document Hyperparameter Decision ⏰ **1 hour**

**Goal**: Either align with spec or justify deviation

**Options**:

**Option A: Match Spec (Conservative)**
```python
# Update configs/stage_0.yaml
lora:
  r: 64  # was 32
  alpha: 64
  target_modules: [...all 7...]

training:
  max_seq_length: 4096  # was 2048
  per_device_train_batch_size: 2  # was 1
  gradient_accumulation_steps: 2  # effective batch = 4
```

**Rationale**: Follow project guidelines for Stage 1 readiness

**Option B: Justify Current (Evidence-Based)**
```markdown
# Hyperparameter Deviation Justification

## Decision: Stay with r=32, max_seq_length=2048

**Evidence**:
- Stage 0 achieved 100% format adherence with r=32
- 2048 tokens sufficient for current problem complexity
- Training on 3-5 examples doesn't require higher capacity

**Risks**:
- May underfit on Stage 1 (50 examples, more complexity)
- Conservative approach would use spec values

**Plan**:
- Keep r=32 for Stage 0 final validation
- Increase to r=64 for Stage 1
- Treat Stage 0 as sensitivity test: "minimal viable capacity"
```

**Deliverable**: `docs/stage0report/hyperparameter_justification.md`

---

## Secondary Actions (Before Stage 1 Starts)

### Add Training Observability ⏰ **4 hours**

**Components**:

**1. Custom Metrics During Training**
```python
class NyayaMetricsCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        # Generate sample on validation prompt
        sample = model.generate(validation_prompt)
        
        # Validate structure
        validator = NyayaStructureValidator()
        result = validator.validate(sample)
        
        # Log to W&B
        wandb.log({
            "val/format_adherence": float(result.has_all_phases),
            "val/phase_count": result.phase_count,
            "val/syllogism_count": result.syllogism_count,
            "val/sample_text": wandb.Html(sample)  # Visual inspection
        })
```

**2. Loss Curves + Learning Dynamics**
- Track: train_loss, val_loss (if validation split exists)
- Learning rate schedule
- Gradient norms
- Token prediction accuracy per phase

**Deliverable**: W&B dashboard with Stage 0 metrics

---

### Benchmark Against Base Model ⏰ **2 hours**

**Goal**: Quantify improvement over untuned baseline

```python
results = {
    "base_model": evaluate(base_model, test_set),
    "tuned_model": evaluate(tuned_model, test_set)
}

comparison = {
    "format_adherence_delta": tuned - base,
    "accuracy_delta": tuned - base,
    "reasoning_quality_delta": tuned - base
}
```

**Expected**:
- Base model: 0% format adherence (no Nyaya structure)
- Tuned model: 100% format adherence
- **Accuracy**: Could go either way (format might help or hurt)

**Deliverable**: `docs/stage0report/baseline_comparison.md`

---

### Document Lessons Learned ⏰ **2 hours**

**Template**:

```markdown
# Stage 0 Lessons Learned

## What Worked
1. Format learnability confirmed with minimal data
2. Corrective iteration process effective (failure → diagnosis → fix → success)
3. DGX Spark + Unsloth pipeline functional

## What Didn't Work
1. Initial hyperparameters too conservative (caused failure)
2. Evaluation pipeline built after training (should be test-first)
3. Only 2 test examples insufficient for statistical confidence

## Critical Insights
1. **Quality over quantity validated**: 3-5 examples sufficient for format
2. **Evaluation-first approach**: Build eval before training, not after
3. **Statistical rigor matters**: 2/2 is not enough, need 10+

## Recommendations for Stage 1
1. Build evaluation infrastructure FIRST
2. Create 10+ held-out examples before training
3. Start with spec-compliant hyperparameters (r=64, not 32)
4. Implement content quality metrics, not just structural
5. Add training observability (custom metrics during training)
```

**Deliverable**: `docs/stage0report/lessons_learned.md`

---

## Stage 0 → Stage 1 Decision Gate

### Success Criteria (Updated)

| Metric | Target | Current | Status | Action |
|--------|--------|---------|--------|--------|
| **Format adherence** | ≥80% | 100% (2/2) | ✅ PASS | Validate with 10 examples |
| **Answer correctness** | ≥60% | 0% exact, ?% semantic | ⚠️ UNKNOWN | Fix evaluator |
| **Content quality** | N/A (not in original spec) | Not measured | ❌ GAP | Implement validators |
| **Statistical confidence** | Confidence interval < 20% | [15.8%, 100%] | ❌ FAIL | Need 10+ examples |
| **Shortcut detection** | Nyaya structure helps | Not tested | ❌ GAP | Run ablation |

### Revised Decision Matrix

**GO for Stage 1 IF**:
1. ✅ Format adherence ≥80% on 10 examples (with CI < 20%)
2. ✅ Answer correctness ≥60% (semantic similarity, not exact match)
3. ✅ Content quality ≥70% (new metric)
4. ✅ Shortcut test passes (Nyaya contributes to accuracy)
5. ✅ Hyperparameter decision documented

**NO-GO (Iterate Stage 0) IF**:
- Format adherence <80% on expanded test set
- Content quality <50% (format learned but content wrong)
- Shortcut detected (model doesn't need structure)

**Current Verdict**: **CONDITIONAL GO**
- ✅ Format learning proven
- ⚠️ Content quality unverified
- ⚠️ Statistical confidence insufficient
- ⚠️ Shortcut risk unmitigated

**Required before Stage 1 starts**: Complete Actions 1–6 above

---

## Timeline (Next 2 Weeks)

### Week 1: Complete Stage 0 Validation

| Day | Actions | Hours | Deliverable |
|-----|---------|-------|-------------|
| Mon | Action 3: Create 8 new test examples | 4 | 10 total held-out examples |
| Tue | Action 2a-2b: Pratyaksha + Udaharana validators | 4 | Content validators (2/4) |
| Wed | Action 2c-2d: Tarka + Hetvabhasa validators | 3 | Content validators (4/4) |
| Thu | Action 1: Fix answer evaluator | 2 | Semantic similarity |
| Fri | Action 4: Run comprehensive validation | 2 | Final metrics with CI |

### Week 2: Verification and Stage 1 Prep

| Day | Actions | Hours | Deliverable |
|-----|---------|-------|-------------|
| Mon | Action 5: Shortcut detection test | 3 | Ablation study |
| Tue | Action 6: Hyperparameter decision | 1 | Justification doc |
| Wed | Secondary: Training observability | 4 | W&B integration |
| Thu | Secondary: Baseline comparison | 2 | Base vs tuned metrics |
| Fri | Secondary: Lessons learned doc | 2 | Stage 0 retrospective |

**Total Effort**: ~30 hours over 2 weeks

**Milestone**: Stage 0 validation complete, Stage 1 GO/NO-GO decision

---

## Risk Assessment

### High Risks

**Risk 1: Content Quality Fails (<50%)**
**Likelihood**: 40%  
**Impact**: High (proves model learned format but not reasoning)  
**Mitigation**: 
- If this happens, Stage 1 needs negative examples (show wrong reasoning)
- May need to strengthen Udaharana teaching in seed examples
- Consider adding explicit content quality loss term in training

**Risk 2: Expanded Test Set Shows Lower Adherence (<80%)**
**Likelihood**: 30%  
**Impact**: Critical (Stage 0 fails, must iterate)  
**Mitigation**:
- If adherence 60-79%: Increase to r=64, max_seq_length=4096, retrain
- If adherence <60%: Revise seed examples, ensure quality
- If still failing: Consider 4-phase simplified structure

**Risk 3: Shortcut Detected**
**Likelihood**: 25%  
**Impact**: High (invalidates Nyaya hypothesis)  
**Mitigation**:
- Analyze WHERE shortcut occurs (format generation but content ignores structure?)
- Add harder problems where shortcuts fail
- Implement constrained decoding to enforce structure use

### Medium Risks

**Risk 4: Answer Semantic Matching Still Low (<60%)**
**Likelihood**: 35%  
**Impact**: Medium (reasoning quality questioned)  
**Mitigation**:
- Manual review: Are answers logically sound but phrased differently?
- If yes: Evaluation issue, not model issue
- If no: Model needs better reasoning examples in Stage 1

**Risk 5: Statistical Noise (10 examples still not enough)**
**Likelihood**: 40%  
**Impact**: Low (proceed with caution to Stage 1)  
**Mitigation**:
- Stage 1 will have 10+ test examples anyway
- Treat Stage 0 as proof-of-concept, not production validation

---

## Appendix: Evidence Trail

All conclusions in this report are based on:

1. **Implementation reports**: `docs/stage0report/*.md`
2. **Training scripts**: `scripts/train_stage0*.py`
3. **Evaluation scripts**: `scripts/evaluate_stage0.py`
4. **Evaluation results**: `results/stage_0_*_evaluation*.json`
5. **Project specifications**: `docs/spec.md`, `docs/pramana_spec_review.md`
6. **Critical reviews**: `docs/stage_0_review.md`, `docs/stage_0_corrective_plan.md`
7. **Infrastructure docs**: Docker setup, Ollama deployment guides

**Traceability**: Every claim in this report can be traced to specific repository artifacts.

---

## Conclusion

Stage 0 has proven the fundamental hypothesis: **Nyaya structure is learnable through fine-tuning**. However, several critical gaps remain before Stage 1:

**Must Complete**:
1. Content quality validators (not just structural)
2. Expanded test set (10 examples, not 2)
3. Semantic answer matching (not exact string)
4. Shortcut detection test (ablation study)

**Timeline**: 2 weeks to complete Stage 0 properly  
**Effort**: ~30 hours  
**Decision Gate**: End of Week 2

**Next Step**: Execute Action 3 (create 8 new test examples) immediately.

---

**END OF REPORT**
