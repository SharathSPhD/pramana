# Pramana Paper Enhancement Tracker

**Purpose**: Collect potential sections, results, comments, and enhancements that would increase the value and impact of the Pramana paper.

**Document Created**: 2026-02-03
**Target Venue**: NeurIPS 2024 or related AI/ML conference
**Current Status**: Draft complete, enhancements in progress

---

## 1. Critical Issues Requiring Resolution

### 1.1 Stage 0 Evaluation Set Ambiguity ⚠️ **HIGH PRIORITY**

**Issue**: Review identified ambiguity in Stage 0 format adherence reporting.

**Current State**:
- Paper references "100% format adherence for Stage 0"
- Two different evaluation results exist:
  - Initial 2-example test set: 100% (2/2) - `stage_0_corrected_evaluation_v7.json`
  - Later 10-example validation: 40% (4/10) - `stage_0_final_validation.json`

**Required Action**:
- [ ] Clarify which evaluation set is the primary result
- [ ] Report both results transparently (e.g., "100% on initial test set, 40% on expanded validation")
- [ ] Add footnote explaining evaluation progression
- [ ] Update Section 7 (Results) to distinguish between test phases

**Draft Language**:
```
Stage 0 achieved 100% format adherence on the initial held-out test set (2 examples)
but showed 40% format adherence when evaluated on an expanded validation set (10 examples),
matching Stage 1's format adherence rate. This suggests that format enforcement remains
challenging even for smaller models when evaluation sets are more diverse.
```

**Files to Update**:
- `sections/07_results.tex` (Stage 0 results subsection)
- `sections/appendices.tex` (detailed evaluation methodology)

---

### 1.2 Z3 Verification Status Clarification ⚠️ **MEDIUM PRIORITY**

**Issue**: Paper discusses Z3 integration but evaluation metrics don't include Z3 verification results.

**Current State**:
- Z3 verifier implemented (`src/pramana/infrastructure/verification/z3_verifier.py`)
- Z3 evaluation handler exists (`src/pramana/application/evaluation/z3_handler.py`)
- Tier 2/3 evaluation (LLM judge, Z3 verification) not run for Stage 0/1
- Review notes this as a transparency issue

**Required Action**:
- [ ] Add explicit statement that Z3 verification is implemented but not yet applied to evaluation
- [ ] Clarify this is planned for Stage 2/3
- [ ] Add Z3 integration to Future Work section with timeline
- [ ] Include Z3 architecture in Implementation section but mark as "not yet evaluated"

**Draft Language**:
```
While we implemented Z3 SMT solver integration for formal logic verification
(see Section 6.3), we did not enable Z3-based metrics in Stage 0/1 evaluations,
which focused on structural validation and semantic correctness. Automated formal
verification is planned for Stage 2 synthetic scaling, where larger datasets will
benefit from automated quality control.
```

**Files to Update**:
- `sections/06_implementation.tex` (add Z3 status note)
- `sections/10_future_work.tex` (Z3 evaluation roadmap)
- `sections/appendices.tex` (Tier 2/3 evaluation methodology)

---

### 1.3 Semantic Correctness Definition ⚠️ **MEDIUM PRIORITY**

**Issue**: Paper reports 100% semantic correctness but doesn't define the evaluation methodology.

**Current State**:
- Semantic correctness evaluated via token overlap (when embeddings disabled)
- Evaluation pipeline in `src/pramana/application/evaluation/pipeline.py`
- Manual judgment for final answers

**Required Action**:
- [ ] Add explicit definition of semantic correctness metric
- [ ] Explain token overlap methodology
- [ ] Document threshold for "correct" (currently binary judgment based on exact match or high overlap)
- [ ] Clarify that this is distinct from format adherence

**Draft Language**:
```
Semantic correctness measures whether the model produces the correct final answer to
the logical problem, regardless of format adherence. For constraint satisfaction and
Boolean SAT problems, we use exact token overlap between the model's extracted answer
and the ground truth. For multi-step deduction problems, we apply a threshold-based
token overlap metric (>80% overlap = correct) combined with manual verification.
```

**Files to Update**:
- `sections/05_methodology.tex` (evaluation metrics subsection)
- `sections/appendices.tex` (detailed metric definitions)

---

## 2. Missing Results and Data

### 2.1 Training Time and Compute Costs 📊 **HIGH VALUE**

**Issue**: Paper mentions budget estimates but doesn't report actual training time/costs.

**Available Data**:
- Training hardware: DGX Spark (A100 40GB GPUs)
- Stage 0: 60 steps, 30 epochs
- Stage 1: 110 steps, 10 epochs
- Can calculate GPU-hours from trainer logs

**Required Action**:
- [ ] Extract training wall-clock time from logs
- [ ] Calculate GPU-hours for both stages
- [ ] Report actual compute costs (if available)
- [ ] Add to reproducibility section

**Data Sources**:
- `models/stage_0_corrected/*/trainer_state.json` (start/end times)
- `models/stage_1/*/trainer_state.json`
- DGX Spark logs (if available)

**Target Location**:
- Section 7 (Results) - Training efficiency subsection
- Appendix - Computational requirements

---

### 2.2 DeepSeek Pre-trained Reasoning Analysis 📊 **HIGH VALUE**

**Issue**: Stage 1 uses DeepSeek-R1-Distill-Llama-8B which has pre-trained reasoning traces, but paper doesn't discuss how this affected results.

**Research Questions**:
- Did DeepSeek's reasoning traces interfere with Nyaya structure learning?
- Did pre-trained reasoning help or hinder format adherence?
- How did base DeepSeek perform vs. base Llama (Stage 0)?

**Required Action**:
- [ ] Add subsection discussing DeepSeek model selection rationale
- [ ] Compare Stage 1 base model outputs vs. Stage 0 base model outputs
- [ ] Discuss interaction between DeepSeek reasoning traces and Nyaya framework
- [ ] Analyze whether DeepSeek's pre-training explains semantic correctness

**Available Evidence**:
- Stage 1 base vs tuned comparison (`docs/figures_stage1_v2/stage1_base_vs_tuned_metrics.csv`)
- Can run additional base model evaluation for comparison

**Target Location**:
- Section 8 (Discussion) - Model selection analysis
- Appendix - Base model comparison

---

### 2.3 Ablation Study Results 📊 **HIGH VALUE**

**Issue**: Review mentions ablation data exists but paper doesn't include it.

**Available Data**:
- `docs/figures_ablation_v1/` directory exists (per review)
- Likely contains LoRA rank, learning rate, epoch count ablations

**Required Action**:
- [ ] Verify ablation data exists and is complete
- [ ] Create ablation results table/figure
- [ ] Add ablation study subsection to Results
- [ ] Discuss impact of hyperparameter choices

**Potential Ablation Dimensions**:
- LoRA rank (32 vs 64 vs 128)
- Learning rate (1e-5 vs 2e-5 vs 5e-5)
- Epoch count (5 vs 10 vs 15 vs 20)
- Training data size (20 vs 35 vs 55 examples)

**Target Location**:
- Section 7 (Results) - Ablation studies subsection
- Appendix - Full ablation results

---

### 2.4 Cross-Stage Progression Analysis 📊 **MEDIUM VALUE**

**Issue**: Paper treats Stage 0 and Stage 1 separately but doesn't deeply analyze progression.

**Available Data**:
- Combined metrics: `docs/figures_combined_v1/stage_combined_metrics.csv`
- Stage 0 avg output length: 3191.8
- Stage 1 avg output length: 3255.2
- Stage 0 format rate: 0.4 (10-example validation)
- Stage 1 format rate: 0.4

**Research Questions**:
- Why did format adherence not improve from Stage 0 to Stage 1?
- Did reasoning quality improve (semantic correctness, justification depth)?
- What changed between stages besides model size?

**Required Action**:
- [ ] Add cross-stage comparison subsection
- [ ] Analyze format adherence stagnation
- [ ] Compare output quality beyond just correctness (verbosity, clarity, structure)
- [ ] Discuss implications for Stage 2 design

**Target Location**:
- Section 8 (Discussion) - Cross-stage analysis subsection

---

### 2.5 Per-Problem-Type Performance Breakdown 📊 **MEDIUM VALUE**

**Issue**: Paper reports aggregate metrics but doesn't break down by problem type.

**Available Data**:
- Problem types: constraint satisfaction, Boolean SAT, transitive reasoning, set operations, deduction
- Evaluation results in `results/stage_1_evaluation.json` have per-example data
- Can categorize by problem type and analyze

**Required Action**:
- [ ] Extract problem type from test file names
- [ ] Calculate format adherence by problem type
- [ ] Calculate semantic correctness by problem type
- [ ] Identify which problem types are most challenging
- [ ] Create table/figure showing breakdown

**Hypothesis**:
- Deduction problems may have lower format adherence (more complex structure)
- Constraint problems may have higher format adherence (simpler structure)

**Target Location**:
- Section 7 (Results) - Problem type analysis subsection
- Appendix - Detailed per-problem results

---

### 2.6 Output Length vs. Format Adherence Correlation 📊 **LOW VALUE**

**Issue**: Could explore whether output verbosity affects format adherence.

**Available Data**:
- Average output lengths in metrics CSV
- Format adherence rates
- Can analyze per-example correlation

**Required Action**:
- [ ] Calculate correlation between output length and format adherence
- [ ] Analyze whether longer outputs have more format errors
- [ ] Create scatter plot if correlation exists

**Target Location**:
- Section 8 (Discussion) - Format failure analysis
- Appendix - Supplementary analyses

---

## 3. Sections Requiring Expansion

### 3.1 Related Work - Epistemic Frameworks in AI 📝 **HIGH VALUE**

**Current State**: Related work covers LLM reasoning, neuro-symbolic AI, verification.

**Missing Content**:
- Other epistemic frameworks applied to AI (Bayesian epistemology, pragmatism, virtue epistemology)
- Formal epistemology in automated reasoning
- Comparison to Western formal logic traditions (Aristotelian syllogism, propositional/predicate logic)
- Why Navya-Nyaya specifically vs. other Indian logical traditions (Buddhist logic, Jain logic)

**Required Action**:
- [ ] Add subsection on epistemic frameworks in AI
- [ ] Discuss why Navya-Nyaya chosen over alternatives
- [ ] Compare to Aristotelian syllogism (similar 5-part structure)
- [ ] Position within broader epistemology research

**Sources to Cite**:
- Matilal (1968, 1998) - Navya-Nyaya vs. Western logic
- Mohanty (1992) - Reason and Tradition in Indian Thought
- Ganeri (2001, 2011) - Semantic Powers, Lost Age of Reason
- BonJour (1985) - Foundationalism vs. coherentism
- Pearl (2000) - Causality and Bayesian epistemology
- Computational epistemology literature

**Target Location**:
- Section 3 (Related Work) - New subsection 3.4

---

### 3.2 Implementation - Training Pipeline Details 📝 **MEDIUM VALUE**

**Current State**: Section 6 covers architecture but light on training pipeline.

**Missing Content**:
- Detailed data preprocessing pipeline
- Training template structure (how instruction/output are formatted)
- Gradient accumulation strategy
- Loss monitoring and early stopping criteria
- Checkpoint selection methodology
- Validation strategy (80/20 split details)

**Required Action**:
- [ ] Add detailed training pipeline diagram
- [ ] Show example of instruction/output format
- [ ] Explain checkpoint selection (best eval loss vs. final)
- [ ] Document data preprocessing steps

**Target Location**:
- Section 6 (Implementation) - Training pipeline subsection

---

### 3.3 Discussion - Failure Mode Analysis 📝 **HIGH VALUE**

**Current State**: Discussion mentions format adherence issues but lacks deep analysis.

**Missing Content**:
- Taxonomy of format failures (missing sections, invalid values, structural errors)
- Root cause analysis for each failure type
- Comparison to similar issues in structured generation literature
- Potential solutions (constrained decoding, parser-in-the-loop, reinforcement learning)

**Required Action**:
- [ ] Create failure taxonomy table
- [ ] Show example of each failure type
- [ ] Analyze whether failures are consistent across examples
- [ ] Propose mitigation strategies for Stage 2

**Available Data**:
- Parse error breakdown: `docs/figures_stage1_v2/stage1_parse_error_breakdown.csv`
- Per-example errors in `results/stage_1_evaluation.json`

**Target Location**:
- Section 8 (Discussion) - Expand failure analysis subsection

---

### 3.4 Future Work - Stage 2/3 Detailed Plans 📝 **MEDIUM VALUE**

**Current State**: Section 10 covers future work but could be more concrete.

**Missing Content**:
- Specific synthetic data generation strategy
- Quality control pipeline for synthetic data
- GRPO training configuration details
- Reward function component definitions
- Evaluation benchmarks for Stage 2/3
- Timeline and resource estimates

**Required Action**:
- [ ] Add detailed Stage 2 plan (synthetic scaling)
- [ ] Add detailed Stage 3 plan (GRPO reinforcement learning)
- [ ] Include reward function equations
- [ ] Specify evaluation benchmarks (LogicBench, ProntoQA, etc.)

**Sources**:
- `CLAUDE.md` Staged Implementation Plan
- `docs/stage_1_comprehensive_report.md` Section 13 (Next Steps)

**Target Location**:
- Section 10 (Future Work) - Expand with detailed roadmap

---

### 3.5 Appendix - Example Outputs and Traces 📝 **HIGH VALUE**

**Current State**: Appendix exists but could include more examples.

**Missing Content**:
- Full Nyaya reasoning traces for multiple examples
- Side-by-side comparison of base vs. tuned outputs
- Example of each failure mode with annotations
- Example of perfect format adherence
- Example showing semantic correctness despite format failure

**Required Action**:
- [ ] Add 3-5 complete example traces to appendix
- [ ] Include one example of each problem type
- [ ] Show base model output vs. tuned model output
- [ ] Annotate examples with phase labels

**Available Data**:
- `docs/stage_1_paper_appendix.md` Section D2 (all examples)
- Can extract from evaluation results

**Target Location**:
- Appendix - Example reasoning traces subsection

---

## 4. Visual Enhancements

### 4.1 Nyaya Framework Flowchart 🎨 **HIGH VALUE**

**Current Need**: Visual representation of 6-phase Nyaya methodology.

**Proposed Figure**:
- Flowchart showing: Problem → Samshaya → Pramana → Pancha Avayava → Tarka → Hetvabhasa → Nirnaya → Answer
- Annotate each phase with brief description
- Show feedback loops (Tarka testing Pancha Avayava, Hetvabhasa invalidating reasoning)

**Implementation**:
- Create Mermaid diagram source
- Convert to PDF for LaTeX
- Add to Section 4 (Nyaya Framework)

**Files to Create**:
- `docs/paper/figures/nyaya_flow.mmd` (Mermaid source)
- `docs/paper/figures/nyaya_flow.pdf` (compiled figure)

**Status**: [ ] Not started

---

### 4.2 Architecture Diagram 🎨 **HIGH VALUE**

**Current Need**: Visual representation of system architecture.

**Proposed Figure**:
- Layered architecture: CLI → Application → Domain → Infrastructure
- Show key components: Parser, Validator, Evaluator, Trainer, Z3 Verifier
- Annotate with file paths

**Implementation**:
- Create Mermaid diagram or draw.io diagram
- Convert to PDF for LaTeX
- Add to Section 6 (Implementation)

**Files to Create**:
- `docs/paper/figures/architecture.mmd` or `.drawio`
- `docs/paper/figures/architecture.pdf`

**Status**: [ ] Not started

---

### 4.3 Problem Type Distribution Chart 🎨 **MEDIUM VALUE**

**Current Need**: Show distribution of training examples by problem type.

**Proposed Figure**:
- Bar chart showing counts by type (constraint, Boolean, transitive, set, deduction)
- Separate bars for Stage 0 vs. Stage 1
- Annotate with percentages

**Implementation**:
- Use matplotlib/seaborn
- Data from seed example counts
- Add to Section 5 (Methodology)

**Files to Create**:
- `docs/figures_combined_v1/problem_type_distribution.png`
- `docs/figures_combined_v1/problem_type_distribution.csv`

**Status**: [ ] Not started

---

### 4.4 Format Failure Taxonomy Diagram 🎨 **MEDIUM VALUE**

**Current Need**: Visual breakdown of format failure types.

**Proposed Figure**:
- Tree diagram or Sankey diagram showing:
  - Parse failures → Missing sections → Hetvabhasa, Nirnaya
  - Parse failures → Invalid values → doubt types, Pramana types
  - Parse failures → Structural errors → malformed markdown

**Implementation**:
- Create tree diagram or Sankey plot
- Data from `stage1_parse_error_breakdown.csv`
- Add to Section 8 (Discussion)

**Files to Create**:
- `docs/figures_stage1_v2/failure_taxonomy.png`

**Status**: [ ] Not started

---

### 4.5 Semantic Correctness vs. Format Adherence Scatter 🎨 **LOW VALUE**

**Current Need**: Visualize the gap between semantic correctness and format adherence.

**Proposed Figure**:
- 2x2 grid: Format correct/incorrect × Semantically correct/incorrect
- Plot each test example as a point
- Most examples should be in "Semantically correct, Format incorrect" quadrant

**Implementation**:
- Scatter plot or 2x2 grid with counts
- Data from `results/stage_1_evaluation.json`
- Add to Section 7 (Results)

**Files to Create**:
- `docs/figures_stage1_v2/correctness_matrix.png`

**Status**: [ ] Not started

---

## 5. Clarifications and Corrections

### 5.1 Stage 1 Seed Example Count 🔍 **LOW PRIORITY**

**Issue**: Review notes 35 Stage 1 seeds in report, 36 files in filesystem.

**Required Action**:
- [ ] Audit `data/seed_examples/stage_one/` directory
- [ ] Identify the extra file (duplicate, template, or unused)
- [ ] Update report or filesystem to match
- [ ] Verify training data matches (55 lines = 20 Stage 0 + 35 Stage 1)

**Files to Check**:
- `data/seed_examples/stage_one/*.md`
- `data/training/stage_1.jsonl`
- `docs/stage_1_comprehensive_report.md` Section 3.1

**Status**: [ ] Not started

---

### 5.2 Evaluation Terminology Consistency 🔍 **MEDIUM PRIORITY**

**Issue**: Paper uses "held-out test set", "validation set", "evaluation set" inconsistently.

**Required Action**:
- [ ] Define terminology clearly:
  - Training set: 80% of examples used for fine-tuning
  - Validation set: 20% of examples held out during training, used for eval loss
  - Test set: Completely separate examples never seen during training
- [ ] Use consistent terminology throughout paper
- [ ] Add terminology glossary to appendix

**Files to Update**:
- All sections (search for "test", "validation", "eval")
- Appendix - Terminology glossary

**Status**: [ ] Not started

---

### 5.3 HuggingFace Repository Links 🔍 **LOW PRIORITY**

**Issue**: Paper mentions HF repos but doesn't provide explicit URLs in some places.

**Required Action**:
- [ ] Add HF repository URLs to Open Source section
- [ ] Verify all links are correct and accessible
- [ ] Add QR codes or short URLs for easy access

**Repositories to Link**:
- Models: `qbz506/nyaya-llama-3b-stage0`, `qbz506/nyaya-llama-3b-stage0-full`
- Models: `qbz506/nyaya-deepseek-8b-stage1`, `qbz506/nyaya-deepseek-8b-stage1-full`
- Datasets: `qbz506/pramana-nyaya-stage0`, `qbz506/pramana-nyaya-stage1`
- Space: `qbz506/pramana-nyaya-demo`

**Files to Update**:
- Section 9 (Open Source)

**Status**: [ ] Not started

---

## 6. Additional Experiments to Run

### 6.1 Base Model Comparison on Nyaya Structure 🧪 **HIGH VALUE**

**Experiment**: Evaluate multiple base models (Llama, DeepSeek, Qwen, Mistral) on Nyaya structured reasoning.

**Motivation**: Understand which model architectures are best suited for structured reasoning.

**Method**:
- Run Stage 0 test set on 4-5 base models (zero-shot)
- Measure format adherence, semantic correctness
- Analyze which phases are most challenging for each model

**Expected Outcome**:
- Identify best base model for Stage 2
- Understand model architecture impact on structured reasoning

**Status**: [ ] Not started

---

### 6.2 Chain-of-Thought vs. Nyaya Comparison 🧪 **HIGH VALUE**

**Experiment**: Compare Nyaya reasoning traces to standard chain-of-thought prompting.

**Motivation**: Demonstrate value of Navya-Nyaya framework over generic reasoning.

**Method**:
- Create CoT prompts for same test problems
- Fine-tune Llama 3.2-3B on CoT data (same size as Nyaya dataset)
- Compare accuracy, output quality, reasoning clarity

**Expected Outcome**:
- Show Nyaya structure improves systematic reasoning
- Quantify advantages of epistemic framework

**Status**: [ ] Not started

---

### 6.3 Human Evaluation Study 🧪 **MEDIUM VALUE**

**Experiment**: Human evaluation of reasoning quality beyond format/semantic correctness.

**Motivation**: Metrics don't capture reasoning clarity, justification quality, epistemic rigor.

**Method**:
- Recruit 3-5 evaluators (ideally with logic background)
- Evaluate 20-30 model outputs on:
  - Reasoning clarity (1-5 scale)
  - Justification adequacy (1-5 scale)
  - Epistemic rigor (appropriate Pramana usage, meaningful Tarka)
  - Overall quality (1-5 scale)
- Calculate inter-rater agreement

**Expected Outcome**:
- Complement automated metrics with human judgment
- Identify quality dimensions missed by automated evaluation

**Status**: [ ] Not started

---

### 6.4 Constrained Decoding Experiment 🧪 **MEDIUM VALUE**

**Experiment**: Apply constrained decoding to enforce format adherence.

**Motivation**: Test whether format issues can be solved with decoding constraints.

**Method**:
- Implement grammar-based constrained decoding (using GBNF or similar)
- Define Nyaya structure grammar
- Re-evaluate Stage 1 model with constrained decoding
- Measure format adherence improvement

**Expected Outcome**:
- Achieve near-100% format adherence
- Determine if constrained decoding hurts semantic correctness

**Status**: [ ] Not started

---

### 6.5 Few-Shot Nyaya Prompting 🧪 **LOW VALUE**

**Experiment**: Test whether few-shot prompting can achieve Nyaya reasoning without fine-tuning.

**Motivation**: Establish whether fine-tuning is necessary or if prompting suffices.

**Method**:
- Create few-shot prompt with 2-3 Nyaya examples
- Test on GPT-4, Claude Opus, Llama 70B
- Compare to fine-tuned Stage 1 model

**Expected Outcome**:
- Show that fine-tuning is superior to prompting for format adherence
- Or discover that few-shot prompting works well (would change project direction)

**Status**: [ ] Not started

---

## 7. Additional Discussion Topics

### 7.1 Epistemic Humility in AI 💭 **HIGH VALUE**

**Topic**: Nirnaya phase requires model to distinguish definitive knowledge from reasonable hypotheses.

**Discussion Points**:
- LLMs typically overconfident in outputs (hallucination problem)
- Navya-Nyaya epistemology explicitly models uncertainty (Samshaya)
- Nirnaya requires acknowledgment when evidence insufficient
- Could Pramana framework reduce overconfidence?

**Evidence to Include**:
- Examples where model correctly states insufficient evidence
- Comparison to base model overconfidence
- Connection to hallucination reduction research

**Target Location**:
- Section 8 (Discussion) - Epistemic implications subsection

**Status**: [ ] Not started

---

### 7.2 Scalability to Real-World Reasoning 💭 **MEDIUM VALUE**

**Topic**: Can Nyaya framework scale beyond formal logic to practical reasoning?

**Discussion Points**:
- Current implementation focuses on logic puzzles
- Real-world reasoning involves uncertain evidence, defeasible reasoning
- Navya-Nyaya designed for practical epistemology, not just formal logic
- Path to expanding beyond constraint satisfaction problems

**Evidence to Include**:
- Examples of real-world reasoning domains (medical diagnosis, legal reasoning, scientific hypothesis testing)
- Discuss how Pramana sources (Pratyaksha, Anumana, Shabda) map to real evidence types
- Challenges in scaling (data generation, evaluation)

**Target Location**:
- Section 8 (Discussion) - Scalability subsection
- Section 10 (Future Work) - Domain expansion

**Status**: [ ] Not started

---

### 7.3 Comparison to Other Structured Reasoning Frameworks 💭 **MEDIUM VALUE**

**Topic**: Position Pramana relative to other structured reasoning approaches.

**Comparison Points**:
- **AlphaProof/AlphaGeometry** (DeepMind): Theorem proving, formal verification
- **Tree-of-Thoughts** (Yao et al. 2023): Branching exploration
- **Self-Consistency** (Wang et al. 2022): Multiple sampling
- **ReAct** (Yao et al. 2023): Reasoning + acting
- **LLEMMA** (Azerbayev et al. 2023): Mathematical reasoning

**Advantages of Pramana**:
- Explicit epistemological grounding
- Integrated fallacy detection
- Counterfactual testing (Tarka)
- Evidence source classification (Pramana)

**Disadvantages**:
- More verbose than alternatives
- Requires extensive training data
- Format adherence challenges

**Target Location**:
- Section 3 (Related Work) - Structured reasoning subsection
- Section 8 (Discussion) - Comparison subsection

**Status**: [ ] Not started

---

### 7.4 Cultural and Philosophical Implications 💭 **LOW VALUE**

**Topic**: Broader implications of applying non-Western epistemology to AI.

**Discussion Points**:
- Western AI research dominated by Western philosophical frameworks
- Indian epistemology offers alternative conceptual tools
- Potential for other non-Western traditions (Chinese, Islamic, African)
- Epistemological pluralism in AI development

**Caveats**:
- Avoid exoticization or cultural essentialism
- Focus on technical value, not cultural nationalism
- Acknowledge scholars who bridged traditions (Matilal, Ganeri, Mohanty)

**Target Location**:
- Section 11 (Conclusion) - Brief mention
- Could be separate position paper

**Status**: [ ] Not started

---

## 8. Writing Quality Improvements

### 8.1 Abstract Strengthening ✍️ **HIGH PRIORITY**

**Current State**: Abstract is clear but could be more compelling.

**Improvements**:
- [ ] Lead with the problem (LLM reasoning failures, hallucination)
- [ ] Emphasize novelty (first application of Navya-Nyaya to LLM training)
- [ ] Highlight key finding (semantic correctness despite format failures)
- [ ] Quantify impact (100% semantic correctness, open-source models)
- [ ] End with broader implications (epistemic frameworks for AI)

**Draft Revision**:
```
Large language models struggle with systematic reasoning, often producing
hallucinated or inconsistent outputs. We introduce Pramana, a novel approach
that teaches LLMs explicit epistemological methodology by fine-tuning on
Navya-Nyaya logic, a 2,500-year-old Indian formal reasoning framework. Unlike
generic chain-of-thought prompting, Navya-Nyaya enforces structured 6-phase
reasoning: doubt analysis, evidence classification, formal syllogism,
counterfactual testing, fallacy detection, and ascertainment. We fine-tune
Llama 3.2-3B and DeepSeek-R1-Distill-Llama-8B models on 55 Nyaya-structured
logical problems, achieving 100% semantic correctness on held-out tests despite
only 40% strict format adherence. This surprising result suggests models learn
reasoning content even when structural enforcement is imperfect. We release all
models, datasets, and code to enable further research on epistemic frameworks
for AI reasoning.
```

**Status**: [ ] Not started

---

### 8.2 Introduction Hook Improvement ✍️ **MEDIUM PRIORITY**

**Current State**: Introduction is thorough but could open more compellingly.

**Improvements**:
- [ ] Start with concrete failure case of LLM reasoning
- [ ] Motivate problem with recent research (Apple GSM-Symbolic study)
- [ ] Introduce Navya-Nyaya as solution more dramatically
- [ ] Clarify contribution more sharply

**Draft Opening**:
```
When OpenAI's o1 model was given the problem "How many r's are in strawberry?",
it produced pages of reasoning before concluding incorrectly. This failure—and
countless others like it—reveals a fundamental gap in large language model
reasoning: statistical pattern-matching without systematic methodology. Recent
research by Apple (Mirzadeh et al. 2024) showed that LLM reasoning performance
degrades 65% when irrelevant context is added, exposing brittle heuristic
reasoning beneath the surface.

We propose a radically different approach: teaching LLMs explicit epistemological
reasoning by fine-tuning on Navya-Nyaya logic, a 2,500-year-old formal framework
that integrates logic, epistemology, and empirical grounding...
```

**Status**: [ ] Not started

---

### 8.3 Transition Smoothing Between Sections ✍️ **LOW PRIORITY**

**Current State**: Section transitions are functional but could flow better.

**Improvements**:
- [ ] Add transition paragraphs between major sections
- [ ] Use forward/backward references more consistently
- [ ] Create narrative arc through paper

**Example Transitions**:
- End of Section 4 (Nyaya Framework) → Beginning of Section 5 (Methodology):
  ```
  Having established the theoretical foundations of Navya-Nyaya reasoning, we
  now turn to the practical challenge of implementing this framework in LLM
  fine-tuning. Section 5 describes our staged training methodology and dataset
  construction strategy.
  ```

**Status**: [ ] Not started

---

## 9. Reproducibility Enhancements

### 9.1 Complete Hyperparameter Table 📋 **HIGH VALUE**

**Current State**: Hyperparameters scattered across methodology section.

**Improvement**: Create comprehensive table with all hyperparameters for both stages.

**Table Contents**:
| Parameter | Stage 0 | Stage 1 | Justification |
|-----------|---------|---------|---------------|
| Base Model | Llama 3.2-3B | DeepSeek-R1-Distill-8B | Size progression |
| LoRA Rank | 64 | 64 | High capacity for complex structure |
| LoRA Alpha | 64 | 64 | Match rank (scaling factor 1.0) |
| Learning Rate | 2e-5 | 2e-5 | Conservative to preserve pre-training |
| Epochs | 30 | 10 | Compensate for data size |
| Batch Size | 2 | 1 | Fit in GPU memory |
| Gradient Accumulation | 4 | 4 | Effective batch size 8 and 4 |
| Sequence Length | 4096 | 4096 | Accommodate full traces |
| Optimizer | AdamW 8-bit | AdamW 8-bit | Memory efficient |
| Precision | bfloat16 | bfloat16 | Stability + efficiency |
| Warmup Steps | 0 | 0 | No warmup needed for small data |
| Weight Decay | 0.01 | 0.01 | Standard regularization |

**Target Location**: Section 5 (Methodology) or Appendix

**Status**: [ ] Not started

---

### 9.2 Dataset Construction Pseudocode 📋 **MEDIUM VALUE**

**Current State**: Dataset construction described in prose.

**Improvement**: Add algorithmic description of data preprocessing.

**Pseudocode**:
```
Algorithm: Generate Nyaya Training Dataset
Input: Markdown examples E = {e₁, e₂, ..., eₙ}
Output: JSONL dataset D

For each example eᵢ in E:
    1. Extract problem statement P from "# Problem" section
    2. Extract reasoning trace R from "## Samshaya" through "## Nirnaya"
    3. Validate R has all 6 required phases
    4. Create training instance: {
         "instruction": P,
         "input": "",
         "output": R
       }
    5. Add to dataset D
Return D
```

**Target Location**: Section 5 (Methodology) or Appendix

**Status**: [ ] Not started

---

### 9.3 Evaluation Pipeline Flowchart 📋 **MEDIUM VALUE**

**Current State**: Evaluation described in text.

**Improvement**: Create flowchart showing evaluation pipeline.

**Flowchart Elements**:
- Input: Test examples
- Model inference
- Parse output (MarkdownParser)
- Validate structure (NyayaStructureValidator)
- Extract answer
- Compare to ground truth
- Calculate metrics (format adherence, semantic correctness)
- Output: Evaluation results

**Target Location**: Section 5 (Methodology) or Appendix

**Status**: [ ] Not started

---

### 9.4 Command Reference Section 📋 **LOW VALUE**

**Current State**: Commands scattered across paper.

**Improvement**: Add appendix section with all reproduction commands.

**Contents**:
- Environment setup (Docker, dependencies)
- Data preparation commands
- Training commands (both stages)
- Evaluation commands
- Model deployment commands (HF upload, Ollama conversion)

**Target Location**: Appendix - Reproduction guide

**Status**: [ ] Not started

---

## 10. Priority Matrix

| Enhancement | Value | Effort | Priority | Status |
|-------------|-------|--------|----------|--------|
| 1.1 Stage 0 eval ambiguity | High | Low | **P0** | Not started |
| 1.2 Z3 status clarification | High | Low | **P0** | Not started |
| 1.3 Semantic correctness definition | High | Low | **P0** | Not started |
| 2.1 Training time/costs | High | Low | **P0** | Not started |
| 2.2 DeepSeek analysis | High | Medium | **P1** | Not started |
| 2.3 Ablation studies | High | Medium | **P1** | Not started |
| 3.1 Epistemic frameworks related work | High | Medium | **P1** | Not started |
| 3.3 Failure mode analysis | High | Low | **P1** | Not started |
| 3.5 Example traces appendix | High | Low | **P1** | Not started |
| 4.1 Nyaya flowchart | High | Medium | **P1** | Not started |
| 4.2 Architecture diagram | High | Medium | **P1** | Not started |
| 8.1 Abstract strengthening | High | Low | **P1** | Not started |
| 9.1 Hyperparameter table | High | Low | **P1** | Not started |
| 2.4 Cross-stage analysis | Medium | Low | **P2** | Not started |
| 2.5 Per-problem breakdown | Medium | Medium | **P2** | Not started |
| 3.2 Training pipeline details | Medium | Medium | **P2** | Not started |
| 3.4 Stage 2/3 detailed plans | Medium | Low | **P2** | Not started |
| 4.3 Problem type distribution | Medium | Low | **P2** | Not started |
| 4.4 Failure taxonomy diagram | Medium | Medium | **P2** | Not started |
| 6.1 Base model comparison | High | High | **P2** | Not started |
| 6.2 CoT vs. Nyaya comparison | High | High | **P2** | Not started |
| 7.1 Epistemic humility discussion | Medium | Low | **P2** | Not started |
| 7.2 Scalability discussion | Medium | Low | **P2** | Not started |
| 7.3 Framework comparison | Medium | Medium | **P2** | Not started |
| 8.2 Introduction hook | Medium | Low | **P2** | Not started |
| 9.2 Dataset pseudocode | Medium | Low | **P2** | Not started |
| 9.3 Evaluation flowchart | Medium | Medium | **P2** | Not started |
| 5.1 Seed example count audit | Low | Low | **P3** | Not started |
| 5.2 Terminology consistency | Low | Low | **P3** | Not started |
| 5.3 HF repository links | Low | Low | **P3** | Not started |
| 2.6 Output length correlation | Low | Low | **P3** | Not started |
| 4.5 Correctness matrix scatter | Low | Low | **P3** | Not started |
| 6.3 Human evaluation | Medium | High | **P3** | Not started |
| 6.4 Constrained decoding | Medium | High | **P3** | Not started |
| 6.5 Few-shot prompting | Low | Medium | **P3** | Not started |
| 7.4 Cultural implications | Low | Low | **P3** | Not started |
| 8.3 Transition smoothing | Low | Low | **P3** | Not started |
| 9.4 Command reference | Low | Medium | **P3** | Not started |

**Priority Definitions**:
- **P0**: Critical for paper quality, must address before submission
- **P1**: High value, should include if time permits
- **P2**: Nice to have, consider for revised version or extended paper
- **P3**: Optional enhancements, low priority

---

## 11. Next Actions

### Immediate Actions (Week 1)
- [ ] Resolve Stage 0 evaluation set ambiguity (Issue 1.1)
- [ ] Add Z3 verification status clarification (Issue 1.2)
- [ ] Define semantic correctness metric explicitly (Issue 1.3)
- [ ] Extract and report training time/costs (Issue 2.1)
- [ ] Strengthen abstract (Issue 8.1)
- [ ] Create comprehensive hyperparameter table (Issue 9.1)

### Short-Term Actions (Week 2-3)
- [ ] Add DeepSeek pre-training analysis (Issue 2.2)
- [ ] Include ablation study results if data exists (Issue 2.3)
- [ ] Expand failure mode analysis (Issue 3.3)
- [ ] Add example reasoning traces to appendix (Issue 3.5)
- [ ] Create Nyaya framework flowchart (Issue 4.1)
- [ ] Create architecture diagram (Issue 4.2)
- [ ] Expand related work on epistemic frameworks (Issue 3.1)

### Medium-Term Actions (Week 4+)
- [ ] Perform cross-stage progression analysis (Issue 2.4)
- [ ] Break down performance by problem type (Issue 2.5)
- [ ] Expand training pipeline details (Issue 3.2)
- [ ] Add detailed Stage 2/3 plans (Issue 3.4)
- [ ] Create additional visualizations (Issues 4.3-4.5)

### Long-Term / Future Paper
- [ ] Run base model comparison experiment (Issue 6.1)
- [ ] Run CoT vs. Nyaya comparison (Issue 6.2)
- [ ] Conduct human evaluation study (Issue 6.3)
- [ ] Test constrained decoding (Issue 6.4)

---

## 12. Contribution Tracking

Use this section to track contributions and updates:

| Date | Contributor | Enhancement | Status |
|------|-------------|-------------|--------|
| 2026-02-03 | Claude Code | Created enhancement tracker | Complete |
| | | | |

---

## 13. Notes and Ideas

### Brainstorm Section

*Use this space for rough ideas and notes that don't fit elsewhere.*

**Potential Additional Sections**:
- Limitations section (currently scattered in Discussion)
- Ethical considerations (minimal currently, could expand)
- Societal impact statement (required by some venues)

**Alternative Experiments**:
- Test Pramana on GSM8K mathematical reasoning benchmark
- Apply to ProntoQA logical reasoning benchmark
- Compare to o1/Claude extended thinking (if we can evaluate)

**Visualization Ideas**:
- Animated GIF showing Nyaya reasoning flow for demo website
- Interactive visualization of parse errors
- Word clouds of common reasoning patterns

**Writing Style**:
- Consider more concrete examples in introduction
- Use running example throughout paper for continuity
- Add "Key Takeaway" boxes for complex sections

---

**Document Status**: Living document, to be updated as enhancements are completed
**Last Updated**: 2026-02-03
**Next Review**: TBD
