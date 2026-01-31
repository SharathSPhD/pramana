# Pramana Project: Technical Specification

**Version**: 1.1 (Revised)
**Date**: 2025-01-30 (Updated based on review feedback)
**Status**: Ready for Stage 0 Implementation

---

## Executive Summary

**Mission**: Prove that LLMs can learn systematic epistemological reasoning via Navya-Nyaya methodology, share trained models on Hugging Face for community research.

**Strategic Approach**:
- Train on DGX Spark (free compute) - NOT cloud GPUs
- Focus on proof-of-concept, NOT production deployment
- Publish models + datasets for community building
- Cost-effective: ~$600-800 total (vs original $12-36K estimate)

**Timeline**: 4.5 months (18 weeks) for Stages 0-2, optional 1 month for Stage 3
**Core Deliverable**: Open-source Nyaya-structured reasoning model enabling reproducible epistemological AI

**Key Innovation**: First LLM fine-tuned on explicit 6-phase Nyaya Darshan methodology, demonstrating that ancient Indian logic can structure modern neural reasoning.

---

## 1. Project Overview & Architecture

### 1.1 Project Vision

**Project Name**: Pramana - Nyaya-Based Epistemic Reasoning Engine

**Core Hypothesis**: Fine-tuning LLMs on systematic 6-phase Nyaya methodology creates better reasoning than generic chain-of-thought by enforcing explicit epistemological structure, comparable to frontier models like o1/Claude extended thinking but based on 2,500-year-old formal logic from Indian philosophy.

**Problem Statement**: Current LLMs suffer from the "Epistemic Gap" - they produce outputs without traceable justification, cannot distinguish belief from knowledge, and hallucinate confident falsehoods. Apple's October 2024 research demonstrated 65% performance degradation when irrelevant context is added, revealing that apparent "reasoning" is often sophisticated pattern-matching.

**Solution Approach**: Apply Navya-Nyaya logic's structured epistemological framework through fine-tuning, creating models that:
- Follow explicit 6-phase reasoning methodology
- Ground claims in valid knowledge sources (Pramanas)
- Construct auditable argument chains (Pancha Avayava)
- Self-verify through counterfactual testing (Tarka)
- Detect and avoid reasoning fallacies (Hetvabhasa)
- Distinguish definitive knowledge from hypothesis (Nirnaya)

**Strategic Mission**: Prove the Pramana paradigm works, share trained models on Hugging Face for community research, enable reproducible epistemological AI. This is a research contribution, not a solo production system.

### 1.2 Architectural Principles

1. **Staged Validation** - Each stage has clear success gates before proceeding
2. **Quality Over Quantity** - 500 high-quality verified examples > 5,000 mediocre ones
3. **Epistemic Humility** - Distinguish Nirnaya (definitive knowledge) from Tarka (hypothesis requiring verification)
4. **Format Enables Reasoning** - 6-phase structure isn't decoration, it's cognitive scaffolding
5. **Incremental Risk Management** - Validate core hypothesis before scaling investment

### 1.3 Technology Stack

**Compute Infrastructure**:
- Platform: NVIDIA DGX Spark
- GPUs: A100 (40GB/80GB)
- Container: Docker-based environment (NVIDIA PyTorch 25.09-py3)

**Machine Learning**:
- Fine-tuning Framework: Unsloth with QLoRA (4-bit quantization)
- Base Model (Primary): DeepSeek-R1-Distill-Llama-8B
- Base Model (Alternative): Qwen 2.5-14B-Instruct
- RL Algorithm: GRPO (Group Relative Policy Optimization)
- Experiment Tracking: Weights & Biases / TensorBoard

**Validation & Verification**:
- Formal Verification: Z3 SMT Solver (optional, for logic subset)
- Quality Control: GPT-4 as LLM judge with explicit Nyaya rubrics
- Data Format: Structured Markdown (YAML frontmatter + markdown sections)

**Deployment** (Stage 4):
- Inference Engine: vLLM for optimized serving
- Load Balancing: Multi-GPU inference
- Monitoring: Custom epistemic quality metrics

### 1.4 Implementation Scope

This specification covers the complete roadmap (Stages 0-4) with detailed implementation focus on Stage 0 (Proof of Concept). Each subsequent stage builds on validated success from the previous stage.

**Stage 0**: Proof of Concept (2 weeks, $100)
**Stage 1**: Minimum Viable Reasoner (8-10 weeks, $500-1000)
**Stage 2**: Synthetic Scaling (8 weeks, $2000-5000)
**Stage 3**: GRPO Enhancement (8-12 weeks, $10,000-30,000)
**Stage 4**: Production Hardening (timeline variable)

---

## 2. The 6-Phase Nyaya Methodology

The Nyaya reasoning framework provides explicit epistemological structure that prevents the "pattern-matching masquerading as reasoning" problem in current LLMs. Each phase has specific computational requirements that translate philosophical rigor into machine-learnable structure.

### 2.1 Phase 1: Samshaya (Doubt Analysis)

**Purpose**: Classify the type of uncertainty before attempting resolution. In Nyaya epistemology, inquiry only begins when there is genuine doubt - forcing the model to articulate what is uncertain prevents jumping to conclusions.

**Doubt Categories**:
1. **Samana Dharma Upapatti**: Multiple entities share properties (most common in constraint satisfaction)
2. **Aneka Dharma Upapatti**: Single entity has multiple conflicting properties
3. **Vipratipatti**: Contradictory testimony from sources
4. **Upalabdhi Avyavastha**: Uncertainty about perception validity
5. **Anupalabdhi Avyavastha**: Uncertainty from absence of evidence

**Computational Requirement**: Model must explicitly identify which category applies and justify why this doubt is worthy of investigation.

**Format**: Markdown section with:
```markdown
## Samshaya (Doubt Analysis)

**Doubt Type**: [Category name]

**Justification**: [Explanation of why this uncertainty exists and why resolution is needed]
```

**Training Signal**: The model learns to pause and categorize uncertainty rather than immediately pattern-matching to likely answers.

### 2.2 Phase 2: Pramana (Evidence Sources)

**Purpose**: Identify valid means of knowledge, preventing hallucination by forcing explicit grounding of all claims.

**Four Pramanas (all required)**:

#### Pratyaksha (Direct Perception)
- **Definition**: Observable facts from the problem statement
- **Constraint**: ONLY verbatim or clear paraphrases from input - no inferences
- **Computational Check**: Can validate by substring matching against problem text
- **Common Error**: Citing inferred facts as "observed"

#### Anumana (Inference)
- **Definition**: Logical deductions with explicit inference type
- **Three Types**:
  - **Purvavat**: Cause → Effect (e.g., smoke implies fire)
  - **Sheshavat**: Effect → Cause (e.g., flood implies prior rain)
  - **Samanyatodrishta**: General correlation (e.g., sun movement implies time passage)
- **Constraint**: Must state which inference type and show the logical connection
- **Common Error**: Treating correlation as causation (Savyabhichara fallacy)

#### Upamana (Comparison)
- **Definition**: Knowledge through analogy to known solved cases
- **Constraint**: Must cite structural similarity to previous examples
- **Use Case**: Case-based reasoning, few-shot learning
- **Common Error**: Superficial metaphors without genuine structural mapping

#### Shabda (Testimony)
- **Definition**: Authoritative logical principles or established rules
- **Examples**: Laws of logic, mathematical axioms, universal principles
- **Constraint**: Must be general principles, not problem-specific facts
- **Common Error**: Restating the problem as a "principle"

**Format**: YAML structure in markdown:
```yaml
### Pratyaksha (Direct Perception)
observable_facts:
  - "Fact 1 (directly stated)"
  - "Fact 2 (directly observable)"

### Anumana (Inference)
inferences:
  - type: purvavat
    premise: "X is true"
    conclusion: "Y must be true"
    justification: "Logical connection"

### Upamana (Comparison)
analogies:
  - reference: "Similar problem type"
    similarity: "Structural mapping"

### Shabda (Authoritative Principles)
principles:
  - "Universal logical rule"
```

**Training Signal**: Model learns to separate observed facts, inferred facts, analogical reasoning, and universal principles - preventing conflation of evidence types.

### 2.3 Phase 3: Pancha Avayava (5-Member Syllogism)

**Purpose**: Construct explicit, auditable reasoning chains. This is the core deductive engine where systematic reasoning occurs.

**Five Required Components (per inference step)**:

1. **Pratijna (Thesis)**: The claim being established
   - Must be specific and testable
   - Example: "Bob has the dog"

2. **Hetu (Reason)**: Evidence supporting the claim
   - Must reference Pramanas from Phase 2
   - Example: "Because constraint 2 directly states this"

3. **Udaharana (Example)**: **Universal rule + concrete instance**
   - **CRITICAL**: Must contain "Wherever X, there is Y" structure
   - NOT just a specific example - must state the general principle
   - Example: "Wherever a direct constraint assigns entity E to position P, there E occupies P. For instance, 'John sits in seat 5' means John is in seat 5."
   - **Common Error**: Providing only specific example without universal rule

4. **Upanaya (Application)**: How universal rule applies to this case
   - Maps the general principle to specific problem
   - Example: "This problem states 'Bob has a dog' as constraint 2"

5. **Nigamana (Conclusion)**: Restated thesis, now justified
   - Should echo Pratijna but now proven
   - Example: "Therefore, Bob has the dog"

**Format**: Markdown sections, multiple Avayava chains for complex problems:
```markdown
### Syllogism 1: [Topic]

**Pratijna (Thesis)**: [Claim]

**Hetu (Reason)**: [Evidence]

**Udaharana (Universal + Example)**: Wherever [general rule], there [consequence]. For example, [concrete instance].

**Upanaya (Application)**: [How rule applies here]

**Nigamana (Conclusion)**: Therefore, [thesis restated]
```

**Training Signal**: Model learns to construct rigorous argument chains with explicit warrants (Udaharana), preventing logical leaps.

### 2.4 Phase 4: Tarka (Counterfactual Testing)

**Purpose**: Verify conclusion via reductio ad absurdum. This is the self-verification mechanism that distinguishes genuine reasoning from lucky guesses.

**Requirements**:
1. Assume the opposite of the conclusion
2. Derive a logical contradiction or absurdity
3. Demonstrate why the negation is impossible
4. NOT just "if X then X" tautology - must test meaningfully

**Format**: Markdown section:
```markdown
## Tarka (Counterfactual Testing)

**Test**: Assume [opposite of conclusion].

Then [logical consequence 1].
But [logical consequence 2].
This contradicts [established fact].

Therefore, [opposite] is impossible, so [original conclusion] must be true.
```

**Training Signal**: Model learns to actively test its conclusions rather than accepting first plausible answer. This is where self-correction capability emerges.

### 2.5 Phase 5: Hetvabhasa (Fallacy Detection)

**Purpose**: Explicit self-audit for reasoning errors. Prevents the model from accepting flawed arguments that "look good" syntactically.

**Five Fallacy Types (all must be checked)**:

1. **Savyabhichara (Erratic Reason)**:
   - Reason correlates with conclusion but doesn't cause it
   - Example: "The ground is wet, therefore it rained" (could be sprinkler)
   - Maps to: Correlation vs. causation errors

2. **Viruddha (Contradictory Reason)**:
   - Reason actually proves the opposite of the conclusion
   - Example: "All ice is cold, this is ice, therefore it's hot"
   - Maps to: Logical contradictions

3. **Prakaranasama (Irrelevant Reason)**:
   - Circular reasoning or off-topic arguments
   - Example: "X is true because X is true"
   - Maps to: Begging the question, circular logic

4. **Sadhyasama (Unproved Reason)**:
   - Premise needs as much proof as the conclusion
   - Example: "Ghosts exist because I saw a ghost"
   - Maps to: Assuming what needs to be proved

5. **Kalaatita (Mistimed Reason)**:
   - Reasoning depends on invalid temporal assumptions
   - Example: Using outdated information as if current
   - Maps to: Temporal logical errors

**Format**: YAML checklist:
```yaml
## Hetvabhasa (Fallacy Detection)

fallacy_checks:
  savyabhichara: none_detected | [description if found]
  viruddha: none_detected | [description if found]
  prakaranasama: none_detected | [description if found]
  sadhyasama: none_detected | [description if found]
  kalaatita: none_detected | [description if found]

reasoning: "Analysis of why no fallacies present, or corrections if found"
```

**Training Signal**: Model learns to be epistemically cautious and self-critical, detecting flaws in its own reasoning.

### 2.6 Phase 6: Nirnaya (Ascertainment)

**Purpose**: Reach definitive conclusion OR explicitly state insufficient evidence. This enforces epistemic humility - the model must distinguish knowledge from hypothesis.

**Two Valid Outcomes**:

1. **Definitive Knowledge (Prama)**:
   - Conclusion survived all tests (Tarka, Hetvabhasa)
   - Answer provided with confidence
   - Status: "Definitive Knowledge"

2. **Epistemic Humility**:
   - Insufficient Pramanas to reach certainty
   - Explicitly state what additional evidence is needed
   - Status: "Hypothesis Requiring Verification"

**Format**: Markdown section:
```markdown
## Nirnaya (Definitive Conclusion)

**Status**: Definitive Knowledge | Hypothesis Requiring Verification

**Answer**: [Final answer if definitive]

**Justification**: [Why this is certain / What evidence is missing]

**Confidence**: [High/Medium/Low with explanation]
```

**Training Signal**: Model learns when to commit to answers vs. when to express uncertainty - preventing hallucinated confidence.

### 2.7 Computational Complexity Analysis

**Token Budget by Phase** (estimated for 4-variable CSP):
- **Samshaya**: 50-100 tokens (doubt classification + justification)
- **Pramana**: 200-400 tokens (4 sources × evidence + structured YAML)
- **Pancha Avayava**: 300-600 tokens (3-5 syllogisms × 120 tokens each)
- **Tarka**: 100-200 tokens (counterfactual test + contradiction)
- **Hetvabhasa**: 150-250 tokens (5 fallacy checks + reasoning)
- **Nirnaya**: 50-100 tokens (conclusion + justification)

**Total**: 850-1,650 tokens (median: ~1,250 tokens)

**Comparison Baseline**:
- GPT-4 standard CoT: 200-400 tokens for same problem
- o1-preview (internal reasoning): 500-800 tokens
- Pramana Nyaya: 1,250 tokens (fully explicated structure)

**Overhead Ratio**: 3-6x vs standard CoT

**Justification**:
The overhead buys **interpretability** and **audit trail**. Similar to formal mathematical proof vs informal argument - longer but verifiable. Each phase serves epistemic function:
- Prevents conflation of evidence types (Pramana separation)
- Forces explicit universal rules (Udaharana "Wherever X")
- Enables error detection (Tarka + Hetvabhasa)
- Distinguishes knowledge from hypothesis (Nirnaya)

For high-stakes reasoning (medical diagnosis, legal arguments, safety-critical systems), 3-6x overhead is acceptable tradeoff for trustworthiness.

**Efficiency Note**: Stage 4 could implement "fast path" for trivial problems (skip full Nyaya) vs. "rigorous path" for complex/critical reasoning.

### 2.8 Phase Quality Dependencies

**Critical Path**: Pramana → Pancha Avayava → Nirnaya

**Dependency Chain**:
- Weak Pramana → Invalid Hetu in Avayava → Wrong conclusion
- Missing Tarka → Can't catch errors in reasoning chain
- Incomplete Hetvabhasa → Fallacies slip through undetected
- Poor Udaharana (no universal rule) → Argument not generalizable

**Phase Quality Thresholds** (for overall solution validity):

| Phase | Minimum Requirement | Score if Failed |
|-------|-------------------|----------------|
| Pramana | All 4 types present with content | 0/10 if any missing |
| Pancha Avayava | ≥2 complete syllogisms with universal rules | 0/10 if <2 valid |
| Tarka | Must test conclusion (not tautological) | 0/10 if circular |
| Hetvabhasa | All 5 fallacy types checked | Partial credit if ≥3 |
| Samshaya & Nirnaya | Structural presence | Pass if present |

**Implication**: A solution can have all 6 phases present but still score poorly if phases are empty template-filling. Quality > format compliance.

---

## 3. Data Format Specification

### 3.1 Format Selection Rationale

**Chosen Format**: Structured Markdown with YAML Frontmatter

**Advantages**:
- Human-readable for manual creation (critical for Stage 0-1)
- Machine-parseable for validation and training
- Git-friendly for version control and collaboration
- Balances structure (YAML metadata) with natural flow (markdown prose)
- Easier to create than pure JSON (no quote escaping, better formatting)

**Rejected Alternatives**:
- Pure JSON: Too mechanical, hard to write manually
- Custom DSL: Adds complexity without clear benefit
- Unstructured text: Can't validate programmatically

### 3.2 File Structure Template

Every training example follows this structure:

```markdown
---
# YAML Frontmatter: Machine-readable metadata
id: pramana-[stage]-[number]
problem_type: constraint_satisfaction | boolean_sat | multi_step_deduction
difficulty: simple | moderate | complex
variables: [number]
ground_truth: "[Expected answer]"
metadata:
  created_date: YYYY-MM-DD
  author: manual | synthetic
  validated: true | false
  z3_verifiable: true | false
  stage: 0 | 1 | 2
---

# Problem

[Natural language problem statement]

**Constraints**:
1. [Constraint 1]
2. [Constraint 2]
...

**Question**: [What needs to be determined]

---

## Samshaya (Doubt Analysis)

**Doubt Type**: [One of 5 categories]

**Justification**: [Why this doubt exists]

---

## Pramana (Evidence Sources)

### Pratyaksha (Direct Perception)
```yaml
observable_facts:
  - "Fact 1 (verbatim or paraphrase from problem)"
  - "Fact 2"
```

### Anumana (Inference)
```yaml
inferences:
  - type: purvavat | sheshavat | samanyatodrishta
    premise: "Starting fact"
    conclusion: "Derived fact"
    justification: "Logical connection"
```

### Upamana (Comparison)
```yaml
analogies:
  - reference: "Similar case or problem type"
    similarity: "Structural mapping explanation"
```

### Shabda (Authoritative Principles)
```yaml
principles:
  - "Universal logical rule or axiom"
```

---

## Pancha Avayava (Systematic Reasoning)

### Syllogism 1: [Topic]

**Pratijna (Thesis)**: [Claim being established]

**Hetu (Reason)**: [Evidence supporting claim]

**Udaharana (Universal + Example)**: Wherever [general rule], there [consequence]. For example, [concrete instance showing rule].

**Upanaya (Application)**: [How universal rule applies to this specific case]

**Nigamana (Conclusion)**: Therefore, [thesis restated as proven]

[Repeat for each reasoning step]

---

## Tarka (Counterfactual Testing)

**Test**: Assume [opposite of conclusion].

[Derivation of contradiction]

Therefore, [original conclusion must be true].

---

## Hetvabhasa (Fallacy Detection)

```yaml
fallacy_checks:
  savyabhichara: none_detected | [description]
  viruddha: none_detected | [description]
  prakaranasama: none_detected | [description]
  sadhyasama: none_detected | [description]
  kalaatita: none_detected | [description]

reasoning: "[Why no fallacies detected OR corrections made]"
```

---

## Nirnaya (Definitive Conclusion)

**Status**: Definitive Knowledge | Hypothesis Requiring Verification

**Answer**: [Final answer]

**Justification**: [Why certain OR what evidence missing]

**Confidence**: [High/Medium/Low with explanation]
```

### 3.3 Validation Schema

Programmatic validation checks:

```python
REQUIRED_SECTIONS = [
    "Problem",
    "Samshaya",
    "Pramana",
    "Pancha Avayava",
    "Tarka",
    "Hetvabhasa",
    "Nirnaya"
]

REQUIRED_PRAMANA_TYPES = [
    "Pratyaksha",
    "Anumana",
    "Upamana",
    "Shabda"
]

REQUIRED_AVAYAVA_COMPONENTS = [
    "Pratijna",
    "Hetu",
    "Udaharana",
    "Upanaya",
    "Nigamana"
]

REQUIRED_HETVABHASA_CHECKS = [
    "savyabhichara",
    "viruddha",
    "prakaranasama",
    "sadhyasama",
    "kalaatita"
]

def validate_nyaya_structure(filepath):
    """Validates that markdown file follows Nyaya structure"""
    content = parse_markdown_with_yaml(filepath)

    # Check YAML frontmatter
    assert "id" in content.metadata
    assert "problem_type" in content.metadata
    assert "ground_truth" in content.metadata

    # Check required sections present
    for section in REQUIRED_SECTIONS:
        assert section in content.sections

    # Check Pramana completeness
    pramana_section = content.sections["Pramana"]
    for pramana_type in REQUIRED_PRAMANA_TYPES:
        assert pramana_type in pramana_section

    # Check Pancha Avayava completeness
    avayava_section = content.sections["Pancha Avayava"]
    syllogisms = extract_syllogisms(avayava_section)
    assert len(syllogisms) > 0
    for syllogism in syllogisms:
        for component in REQUIRED_AVAYAVA_COMPONENTS:
            assert component in syllogism
        # Check Udaharana has universal rule
        assert "Wherever" in syllogism["Udaharana"]

    # Check Hetvabhasa completeness
    hetvabhasa_section = content.sections["Hetvabhasa"]
    for fallacy_type in REQUIRED_HETVABHASA_CHECKS:
        assert fallacy_type in hetvabhasa_section

    return True
```

### 3.4 Data Versioning Strategy

**Git Structure**:
```
data/seed_examples/
├── .dataversion          # Version metadata
├── stage_zero/
│   └── v1.0/            # Immutable after Stage 0 complete
├── stage_one/
│   ├── v1.0/            # Initial 50 examples
│   ├── v1.1/            # Refinements after first training
│   └── v2.0/            # Format changes (BREAKING)
└── stage_two_synthetic/
    └── v1.0/
```

**Metadata Tracking** (`.dataversion` file):
```yaml
version: 1.1
created: 2025-01-30
stage: 1
examples_count: 50
quality_scores:
  mean_tier2_score: 0.87
  manual_review_pass_rate: 0.92
git_commit: abc123def
changes:
  - "Improved Udaharana universal rules in 12 examples"
  - "Fixed Tarka tautology issues in 5 examples"
```

**Breaking Changes** (v1.0 → v2.0):
- YAML schema modifications
- Phase additions/removals
- Requires model retraining from scratch
- Backward compatibility not maintained

**Version Control Strategy**:
- **Immutable versions**: Once training starts, that version is frozen
- **Iterative refinement**: Create v1.1 for improvements, retrain and compare
- **Git tags**: Tag each version (e.g., `data-v1.0`) for reproducibility

### 3.5 Example Quality Lifecycle

**Quality Tiers**:
- **Gold** (Tier2 score ≥0.90): Permanent, never remove, use in all future training
- **Silver** (0.80-0.89): Review after Stage 2, possibly refine or replace
- **Bronze** (<0.80): Candidate for removal if dataset >100 examples

**Retirement Criteria**:
- Model consistently ignores example (attention weight analysis shows <0.1 weight)
- Contains identified Nyaya methodology errors discovered post-creation
- Superseded by higher-quality version of same problem type

**Retirement Process**:
1. **Don't delete** - move to `data/archived/retired_YYYY-MM-DD/`
2. Update `.dataversion` with retirement reason and replacement ID
3. Track impact on model performance after removal (A/B test)
4. Document lessons learned for future example creation

---

## 4. Stage 0: Proof of Concept Implementation

**Timeline**: 2 weeks
**Budget**: $100
**Goal**: Validate that LLMs can learn the 6-phase Nyaya structure

### 4.1 Success Criteria

**Understanding Stage 0 Overfitting** (Why It's Actually Good):

With 4 training examples, the model will achieve ~100% training accuracy through **complete memorization**. This is INTENTIONAL and DESIRABLE because:

1. **Hypothesis Test**: We're testing "Can the model learn THIS structure?" not "Can it generalize?"

2. **Memorization IS Learning**: If the model memorizes the structure correctly, it proves the 6-phase format is learnable by the architecture. If it can't memorize even after 10 epochs, the format is too complex for the model size.

3. **Held-Out Test Goal**: The Zebra puzzle validation isn't about accuracy (that would be random chance). It's about: "Does the model ATTEMPT Nyaya structure on a new problem, or does it abandon it?"

4. **Acceptable Outcomes Hierarchy**:
   - ✅ **Best**: Model overfits training, applies structure to validation, gets answer right
   - ✅ **Good**: Model overfits training, applies structure to validation, answer wrong but reasoning valid
   - ⚠️ **Concerning**: Model overfits training, gets validation right via non-Nyaya shortcut
   - ❌ **Failure**: Model can't overfit training even after 10 epochs
   - ❌ **Failure**: Model ignores structure on validation, produces generic CoT

**Validation Goal**: 80% structural adherence at 60% accuracy means "model is TRYING to use Nyaya even when it fails" which validates the learnability hypothesis.

**Primary Success Metric**: 80%+ structural completeness on held-out Zebra puzzle
- All 6 phases present and in correct order
- Each phase contains required components
- Model attempts systematic reasoning (even if answer incorrect)

**Secondary Success Metrics**:
- 60%+ answer correctness on validation problem
- ≥3 valid Pancha Avayava syllogisms per solution
- Zero instances of completely abandoning structure mid-solution
- Manual review confirms phases contain appropriate content (not template-filling)

**Failure Criteria** (triggers reassessment):
- <50% structural adherence → Base model can't learn format
- Generic chain-of-thought instead of Nyaya → Prompt issue
- Correct answers without Nyaya structure → Shortcut detected (see Section 4.5.1)

### 4.2 Seed Example Creation (Week 1, Days 1-5)

**Manual Creation of 5 Examples**

Following structured complexity progression:

**Problem 1: Three Variables, Direct Constraints** (~2 hours)
- Purpose: Establish baseline - can model follow 6 phases at all?
- Example: Alice/Bob/Carol with cat/dog/fish
- Constraints: Simple direct assignments and negations
- Teaches: Basic Pratyaksha identification, simple Anumana elimination

**Problem 2: Three Variables, Relational Constraints** (~2.5 hours)
- Purpose: Introduce comparative reasoning
- Example: Dan/Eve/Frank race finishing positions
- Constraints: "Before/after" relationships
- Teaches: Samanyatodrishta Anumana (transitive inference), Upamana usage

**Problem 3: Four Variables, Mixed Constraints** (~3 hours)
- Purpose: Test if structure scales to harder problems
- Example: Four houses with colors, spatial reasoning
- Constraints: Mix of spatial, positive, and negative
- Teaches: Multiple Pancha Avayava chains, sequential elimination

**Problem 4: Four Variables, Negative Constraints** (~3 hours)
- Purpose: Test Hetvabhasa detection capabilities
- Example: Four friends speaking languages (all negative constraints)
- Constraints: Only "not X" statements
- Teaches: Systematic elimination when only exclusions given, avoiding circular reasoning

**Problem 5: Classic Zebra Puzzle (5×5)** (~4 hours)
- Purpose: Integration test for full methodology
- Problem: Einstein's Zebra puzzle with 14 constraints
- Purpose: Prove methodology works on genuinely difficult problems
- Teaches: Complex multi-step reasoning, discipline vs. trial-and-error

**Total Time**: ~15 hours manual creation

**Deliverable**: 5 markdown files in `data/seed_examples/stage_zero/`

**Quality Gate Checklist** (per example):
- [ ] All 6 phases present and in order
- [ ] Pratyaksha contains ONLY observable facts from problem
- [ ] Each Udaharana contains "Wherever X, there is Y" universal rule
- [ ] Tarka actually tests conclusion via reductio ad absurdum
- [ ] All 5 Hetvabhasa types explicitly checked
- [ ] Ground truth answer is verifiable and correct
- [ ] Natural language flows readably (not just template-filling)

### 4.3 Training Infrastructure Setup (Week 1, Days 3-5, parallel with example creation)

**Docker Environment Setup**:
```bash
# Execute automated setup
cd ~/pramana-project
bash pramana_docker_setup.sh

# Verify installation
./run_pramana_container.sh
python scripts/test_unsloth.py
```

**Training Configuration** (`configs/stage_zero_config.yaml`):
```yaml
experiment_name: "pramana_stage_zero"
stage: 0

model:
  base_model: "unsloth/Meta-Llama-3.1-8B-bnb-4bit"
  load_in_4bit: true
  lora_config:
    r: 64  # High rank for learning new paradigm
    lora_alpha: 16
    target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    lora_dropout: 0.05
    bias: "none"

data:
  seed_examples_path: "/workspace/pramana/data/seed_examples/stage_zero"
  num_examples: 5
  train_test_split: 0.8  # 4 train, 1 validation

training:
  output_dir: "/workspace/pramana/models/checkpoints/stage_zero"
  num_train_epochs: 10  # Embedding new reasoning structure
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 4  # Effective batch = 8
  learning_rate: 2.0e-4
  warmup_steps: 10
  logging_steps: 1
  save_steps: 50
  max_seq_length: 4096  # Long reasoning traces

evaluation:
  metrics: ["format_adherence", "reasoning_completeness", "answer_accuracy"]
  validation_examples: 1  # Zebra puzzle held out
```

**Data Pipeline Script** (`scripts/data/parse_markdown_examples.py`):
```python
def parse_nyaya_markdown_to_training_format(file_path):
    """Convert markdown Nyaya example to training conversation format"""

    # Parse YAML frontmatter and markdown sections
    with open(file_path) as f:
        content = frontmatter.load(f)

    # Extract problem statement
    problem_text = extract_section(content.content, "Problem")

    # Extract full Nyaya solution
    nyaya_solution = ""
    for section in ["Samshaya", "Pramana", "Pancha Avayava",
                    "Tarka", "Hetvabhasa", "Nirnaya"]:
        nyaya_solution += extract_section(content.content, section)

    # Format as conversation for training
    conversation = {
        "conversations": [
            {
                "role": "system",
                "content": "You are a reasoning engine that solves logical problems using Nyaya Darshan methodology. Apply the systematic six-phase approach."
            },
            {
                "role": "user",
                "content": problem_text
            },
            {
                "role": "assistant",
                "content": nyaya_solution
            }
        ]
    }

    return conversation
```

### 4.4 Training Execution (Week 2, Days 1-3)

**Data Split**:
- Training: Problems 1-4 (simple → complex progression)
- Validation: Problem 5 (Zebra puzzle - integration test)

**Training Command**:
```bash
python scripts/training/stage_zero_finetune.py \
  --config configs/stage_zero_config.yaml \
  --seed_examples data/seed_examples/stage_zero/ \
  --output_dir models/checkpoints/stage_zero \
  --wandb_project pramana-stage-zero \
  --experiment_name "stage0-llama31-8b-lora64"
```

**Expected Training Behavior**:
- Training time: 2-4 hours on single A100
- Training loss: Should decrease to 0.5-1.0 (massive overfitting expected with 4 examples)
- Perplexity tracking: Monitor which phases are harder to learn
- Sample generations: Check every 100 steps to see structure emerging

**Monitoring Checklist**:
- [ ] Training loss decreasing consistently
- [ ] Sample outputs showing phase structure (even if content wrong)
- [ ] Model not degenerating to gibberish or repetition
- [ ] GPU utilization stable at 80-90%
- [ ] No OOM errors

### 4.5 Evaluation (Week 2, Days 4-5)

**Evaluation Script** (`scripts/evaluation/stage_zero_eval.py`):
```python
def evaluate_stage_zero(model, tokenizer, validation_problem_path):
    """Test model on held-out Zebra puzzle"""

    # Load validation problem
    problem = load_nyaya_markdown(validation_problem_path)
    problem_text = extract_section(problem.content, "Problem")
    ground_truth = problem.metadata["ground_truth"]

    # Generate solution (multiple samples for analysis)
    samples = []
    for i in range(5):  # Generate 5 solutions
        output = model.generate(
            problem_text,
            max_new_tokens=3000,
            temperature=0.7,
            do_sample=True
        )
        samples.append(output)

    # Tier 1: Structural Validation
    structure_results = []
    for sample in samples:
        score = validate_nyaya_structure(sample)
        structure_results.append(score)

    # Calculate structural completeness rate
    completeness_rate = sum(r["has_all_phases"] for r in structure_results) / len(samples)

    # Tier 2: Content Quality (Manual Review)
    print("=== MANUAL REVIEW REQUIRED ===")
    for i, sample in enumerate(samples):
        print(f"\n--- Sample {i+1} ---")
        print(sample)
        print("\nQuality Checklist:")
        print("1. Samshaya: Appropriate doubt type? (Y/N)")
        print("2. Pratyaksha: Only observables? (Y/N)")
        print("3. Udaharana: Has universal rule? (Y/N)")
        print("4. Tarka: Tests conclusion? (Y/N)")
        print("5. Hetvabhasa: All checked? (Y/N)")
        print("6. Answer: Correct? (Y/N)")

        quality_scores = manual_input_scores()  # Interactive scoring

    # Tier 3: Answer Correctness
    correct_answers = sum(
        extract_answer(sample) == ground_truth
        for sample in samples
    )
    accuracy_rate = correct_answers / len(samples)

    # Generate Report
    report = {
        "structural_completeness": completeness_rate,
        "answer_accuracy": accuracy_rate,
        "structure_by_sample": structure_results,
        "manual_quality_scores": quality_scores,
        "success_criteria_met": {
            "structure_80_percent": completeness_rate >= 0.80,
            "accuracy_60_percent": accuracy_rate >= 0.60,
            "valid_syllogisms": check_syllogism_count(samples)
        }
    }

    return report
```

**Success Decision Matrix**:

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Structural completeness | ≥80% | ___% | PASS/FAIL |
| Answer accuracy | ≥60% | ___% | PASS/FAIL |
| Valid syllogisms per solution | ≥3 | ___ | PASS/FAIL |
| Structure abandonment | 0% | ___% | PASS/FAIL |

**Outcome Decisions**:

✅ **ALL PASS → Proceed to Stage 1**
- Core hypothesis validated
- Model can learn Nyaya structure
- Ready for 50-example expansion

⚠️ **PARTIAL PASS → Iterate**
- Structure good but accuracy poor: Need more training epochs or better base model
- Accuracy good but structure poor: Improve data quality or prompt format
- Some phases weak: Focus additional examples on weak areas

❌ **FAIL → Reassess Approach**
- <50% structure: Format may be too complex, try simpler 4-phase version
- Generic CoT output: Training data not distinctive enough, increase contrast
- Correct without structure: Model shortcutting, need more complex problems

### 4.5.1 The "Shortcut Detection Test" (CRITICAL)

**Failure Mode**: Model gets correct answer on Zebra puzzle BUT:
- Skips most Nyaya phases or fills them with gibberish
- Uses standard constraint propagation reasoning internally
- Happens to format output with phase headers as decoration
- Structure is cosmetic, not functional

**Detection Method** (Ablation Test):

1. **Baseline**: Evaluate model with full Nyaya prompt instructions
   - Measure: Accuracy on validation set

2. **Ablation**: Remove format instructions, give only problem statement
   - Measure: Accuracy on same validation set

3. **Analysis**:
   - If accuracy STAYS THE SAME (±5%) → **Model found shortcut**, not using Nyaya
   - If accuracy DROPS significantly (>15%) → **Structure genuinely helps reasoning**

**Implementation**:
```python
# Test 1: Full Nyaya prompt
prompt_nyaya = """Solve using 6-phase Nyaya methodology:
Samshaya, Pramana, Pancha Avayava, Tarka, Hetvabhasa, Nirnaya"""

accuracy_with_structure = evaluate(model, validation_set, prompt_nyaya)

# Test 2: Minimal prompt (no format instructions)
prompt_minimal = """Solve this logic puzzle and explain your reasoning:"""

accuracy_without_structure = evaluate(model, validation_set, prompt_minimal)

# Verdict
if abs(accuracy_with_structure - accuracy_without_structure) < 0.05:
    print("❌ SHORTCUT DETECTED: Model doesn't need Nyaya structure")
else:
    print("✅ STRUCTURE HELPS: Nyaya methodology contributes to reasoning")
```

**If Shortcut Detected**:
- **Root Cause**: Problem selection was too easy; model can solve CSPs without systematic methodology
- **Fix Option 1**: Create harder problems where trial-and-error fails (6-variable CSP, 25+ constraints)
- **Fix Option 2**: Accept that for "easy" problems, Nyaya overhead isn't needed (design fast path for Stage 4)
- **Fix Option 3**: Change problem types to domains where shortcuts don't exist (mathematical proofs, causal reasoning)

**Decision**: If shortcut is fundamental (model always finds it), pivot to **hybrid approach**: Simple problems use shortcut, complex problems use full Nyaya.

### 4.6 Stage 0 Deliverables

**Artifacts**:
1. ✅ 5 gold-standard seed examples (markdown format)
2. ✅ Trained model checkpoint (`models/checkpoints/stage_zero/final/`)
3. ✅ Evaluation report with metrics and sample outputs
4. ✅ Lessons learned document:
   - What worked: Which phases were easiest to learn?
   - What struggled: Where did model deviate from structure?
   - Recommendations: How to improve for Stage 1?

**Cost Breakdown**:
- Manual effort: 15-20 hours (example creation + evaluation)
- Compute: $30-50 (DGX Spark A100 time, 4-6 hours)
- Tools: $0 (all open source)
- **Total: ~$50-100**

**Timeline Checkpoint**: End of Week 2
- If successful: Begin Stage 1 planning
- If needs iteration: Allow 1 additional week for refinement
- If failed: Conduct postmortem and reassess hypothesis

### 4.7 Stage 0 Failure Recovery Plan

**Scenario 1: <50% Structural Adherence**

**Symptoms**:
- Model generates text but doesn't follow phase structure
- Phases appear out of order or missing
- Format completely broken

**Root Cause Analysis**:
- Format too complex for 5 examples with 8B model
- Base model architecture incompatible with structured generation
- Training hyperparameters wrong (learning rate too high, epochs too few)

**Recovery Actions**:
1. **Simplify structure**: Reduce to 4 phases (Pramana, Avayava, Tarka, Nirnaya)
2. **Try different base model**: Switch DeepSeek ↔ Qwen ↔ Llama 3.1
3. **Adjust training**: Increase epochs to 15, reduce learning rate to 1e-4
4. **Add format examples**: Include 2-3 additional training examples focusing purely on structure

**Timeline**: +1 week to recreate simplified examples and retrain

**Kill Criteria**: If all 3 base models fail with simplified 4-phase structure after fixes

---

**Scenario 2: Generic Chain-of-Thought Output**

**Symptoms**:
- Model produces reasonable reasoning but ignores Nyaya format
- Output looks like standard GPT-4 CoT
- Phase headers might appear but content is generic

**Root Cause Analysis**:
- Training data not distinctive enough from base model's pre-training
- Model defaulting to familiar CoT patterns
- Nyaya terminology not emphasized enough

**Recovery Actions**:
1. **Enhance distinctiveness**: Add more Nyaya-specific terminology (Vyapti, Drishtanta, etc.)
2. **Contrastive examples**: Create pairs showing generic CoT vs Nyaya reasoning side-by-side
3. **Stronger system prompts**: Emphasize "You MUST use Nyaya Darshan methodology explicitly"
4. **Check data quality**: Ensure seed examples themselves are distinctive, not just formatted CoT

**Timeline**: +1 week to enhance examples with distinctive Nyaya content

**Kill Criteria**: If model still produces generic CoT after enhancement across all 3 base models

---

**Scenario 3: Correct Answers Without Structure (Shortcut)**

**Symptoms**:
- Model gets Zebra puzzle correct
- But uses non-Nyaya reasoning or minimal structure
- Ablation test shows no accuracy drop without format

**Root Cause Analysis**:
- Problems too easy; model can solve with pattern matching
- Nyaya structure not necessary for constraint satisfaction
- Base model already has strong CSP-solving capabilities

**Recovery Actions**:
1. **Harder problems**: Replace with 6-variable CSPs, 25+ constraints
2. **Different domains**: Shift to domains where shortcuts don't exist:
   - Mathematical theorem proving (requires explicit justification)
   - Causal reasoning (requires distinguishing correlation from causation)
   - Adversarial logic (contradictory premises requiring Hetvabhasa detection)
3. **Accept hybrid**: Simple problems don't need Nyaya; focus on complex reasoning

**Timeline**: +2 weeks to create harder examples in new domains

**Kill Criteria**: If model finds shortcuts in ALL problem types (CSP, SAT, proofs, causal)

---

**Scenario 4: Complete Training Failure**

**Symptoms**:
- Training loss doesn't decrease
- Model outputs gibberish or repetitive text
- GPU memory errors or NaN losses

**Root Cause Analysis**:
- Technical issue: LoRA configuration wrong, batch size too large
- Data issue: Examples corrupted or unparseable
- Model issue: Base model checkpoint corrupted

**Recovery Actions**:
1. **Verify environment**: Re-run `test_unsloth.py`, check CUDA version
2. **Reduce complexity**: Batch size = 1, gradient accumulation = 1, LoRA rank = 16
3. **Test with tiny model**: Try Llama 3.2 1B first to isolate issue
4. **Check data**: Validate all examples parse correctly, no encoding issues

**Timeline**: +3 days to debug and fix technical issues

**Kill Criteria**: If hardware failure (DGX Spark unavailable) → Switch to cloud GPU backup

---

**Ultimate Kill Criteria** (Abandon Project):

If ALL of the following are true after fixes:
1. ❌ All 3 base models (DeepSeek, Qwen, Llama) fail
2. ❌ Simplified 4-phase structure still fails
3. ❌ Multiple problem domains attempted (CSP, SAT, proofs, causal)
4. ❌ Technical issues ruled out (training works on toy examples)

**Then**: Write paper about **why Nyaya-LLM approach failed**, publish negative results, pivot to:
- Simpler approach: Single-phase "show your work" without full Nyaya
- Different philosophy: Try Mimamsa (deontic logic) or Buddhist logic
- Accept limitation: Some epistemological structures don't map to neural architectures

**Value Even if Failed**: Negative results publishable, methodology documented for future researchers

---

## 5. Stage 1-3 Implementation Roadmap

**Strategic Focus**: All training on DGX Spark (free compute), prove Pramana paradigm, share models on Hugging Face for community research. Stage 3 (GRPO) is OPTIONAL - supervised fine-tuning may be sufficient.

### 5.1 Stage 1: Minimum Viable Reasoner (8-10 weeks, ~$200)

**Goal**: Build production-ready training dataset, achieve 90%+ format adherence with 60-70% accuracy, publish first model to Hugging Face

**Seed Example Creation** (Weeks 3-6):

**Total**: 50 gold-standard examples

**Problem Distribution**:
- **25 Constraint Satisfaction** (50% of dataset)
  - 10 three-variable problems (establish baseline patterns)
  - 10 four-variable problems (standard difficulty)
  - 5 five-variable Zebra-style (complex reasoning)

- **15 Boolean SAT** (30% of dataset)
  - 5 simple 3-clause problems
  - 7 moderate 5-7 clause problems
  - 3 complex 10+ clause problems
  - Teaches propositional logic with Nyaya structure

- **10 Multi-step Deduction** (20% of dataset)
  - Syllogistic reasoning: "All X are Y, Some Y are Z..."
  - Validates Pancha Avayava on formal logic
  - Bridge to broader reasoning domains

**+ 5 Negative Examples** (Contrastive Learning):

Create 5 INTENTIONALLY FLAWED examples demonstrating common errors:

1. **Pratyaksha Contamination**: Pratyaksha includes inferred facts (teaches: only observables allowed)
2. **Missing Universal Rule**: Udaharana with specific example but no "Wherever X, there is Y" structure (teaches: universal rule required)
3. **Circular Tarka**: Tarka that's tautological/doesn't actually test conclusion (teaches: must genuinely test)
4. **Incomplete Hetvabhasa**: Missing fallacy checks (teaches: all 5 types required)
5. **False Certainty**: Nirnaya claiming definitive knowledge without proper Pramana grounding (teaches: epistemic humility)

**Format**: Label as `negative_example: true` in YAML frontmatter
**Training Approach**:
- Use contrastive learning pairs (good example vs flawed version)
- Or DPO-style preference training (model learns to prefer correct structure)
**Validation**: Model should score these examples lower in Tier 2 evaluation

**Total Stage 1**: 55 examples (50 positive + 5 negative)

**Creation Timeline**:
- Week 3: 12-13 examples (mix of CSP and SAT)
- Week 4: 12-13 examples
- Week 5: 12-13 examples
- Week 6: Remaining examples + quality review

**Training Approach** (Week 7-8):

**Model Selection**:
- Primary: DeepSeek-R1-Distill-Llama-8B (has pre-trained reasoning traces)
- Alternative: Qwen 2.5-14B-Instruct (strong logic capabilities)
- Test both in parallel if compute allows

**LoRA Configuration**:
```yaml
lora_config:
  r: 64-128  # High rank for complex reasoning paradigm
  lora_alpha: 64-128  # Match rank
  target_modules:
    - q_proj, k_proj, v_proj, o_proj  # Attention
    - gate_proj, up_proj, down_proj   # FFN
    - embed_tokens, lm_head            # Input/output
  lora_dropout: 0.05
  bias: "none"
```

**Training Hyperparameters**:
```yaml
training:
  num_train_epochs: 10-15  # Embedding new paradigm
  learning_rate: 2e-5      # Conservative to preserve reasoning
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16  # Effective batch = 16
  max_seq_length: 4096
  warmup_ratio: 0.1
  lr_scheduler_type: "cosine"
  weight_decay: 0.01
  max_grad_norm: 0.3
```

**Evaluation** (Weeks 9-10):

**Format Adherence**: Target >90%
- All 6 phases present and ordered
- Components within phases complete
- Measured on 10 held-out examples

**Structure-Accuracy Correlation**:
```python
correlation_metrics = {
    "problems_with_full_structure": {
        "count": X,
        "accuracy": Y%
    },
    "problems_missing_tarka": {
        "count": X,
        "accuracy": Y%  # Should be lower than full structure
    },
    "problems_incomplete_pramana": {
        "count": X,
        "accuracy": Y%
    }
}
```

**Critical Question**: Is accuracy positively correlated with structural completeness? If not, the Nyaya structure isn't helping reasoning.

**Benchmark Introduction**:
- ProntoQA (logical inference, ontological reasoning)
- Simple RuleTaker examples (rule-based deduction)
- Custom Nyaya test set (10 new problems)

**Not expecting to beat baselines yet** - measuring whether Nyaya methodology transfers to unseen problems.

**Success Criteria**:
- ✅ 90%+ format adherence on held-out problems
- ✅ 60-70% accuracy (better than random, establishing baseline)
- ✅ Positive correlation: complete structure → higher accuracy
- ✅ Generalizes to ≥2 problem types (e.g., CSP + Boolean SAT)

**Deliverables**:
- 50 validated seed examples
- Trained model checkpoint with evaluation
- Structure-accuracy correlation analysis
- **Hugging Face model release**: `pramana/nyaya-llama-8b-stage1`
- Recommendations for Stage 2 synthetic generation

**Hugging Face Release Strategy**:
```bash
# Push model to Hugging Face
huggingface-cli login
huggingface-cli upload pramana/nyaya-llama-8b-stage1 ./models/checkpoints/stage_one/final

# Include in model card:
# - All 50 training examples (in examples/ directory)
# - Evaluation metrics and benchmark results
# - Inference code and usage examples
# - Citation info and research context
```

**Cost Breakdown**: ~$200 total
- Manual effort: 60-80 hours (example creation + evaluation)
- Compute: **$0** (DGX Spark)
- LLM API calls: $100-200 (GPT-4 for evaluation/scoring if needed)
- Tools: $0 (all open source)

**Community Value**: First open Nyaya-structured reasoning model, enables reproduction and extensions

---

### 5.2 Stage 2: Synthetic Scaling (8 weeks, ~$300-500)

**Goal**: Scale to 200-500 verified examples using three-tier quality control

**Three-Tier Quality Control Pipeline**:

**Tier 1: Automated Structural Filters** (100% of generated examples)

```python
def tier1_structural_validation(example):
    """Fast automated checks - reject immediately if fails"""

    filters = {
        "valid_json_schema": validate_yaml_frontmatter(example),
        "has_all_phases": check_six_phases_present(example),
        "correct_phase_order": check_sequential_order(example),
        "pramana_complete": all_four_pramanas_present(example),
        "avayava_complete": five_components_present(example),
        "z3_verifiable": verify_with_z3(example) if is_formal_logic(example) else True,
        "has_answer": answer_in_nirnaya(example),
        "reasonable_length": 100 < len(str(example)) < 10000,
        "no_self_contradiction": not has_internal_contradiction(example)
    }

    return all(filters.values()), filters
```

**Pass Rate Target**: 70-80% (if lower, generation prompts need fixing)

**Tier 2: LLM-as-Judge Nyaya Quality Scoring** (100% of Tier 1 passes)

Uses GPT-4 with explicit Nyaya rubric to score on 0-10 scale:
- Samshaya appropriateness
- Pratyaksha validity (only observables?)
- Anumana correctness (actual inferences?)
- Upamana relevance (appropriate analogies?)
- Shabda correctness (valid principles?)
- Pancha Avayava quality (universal rules in Udaharana?)
- Tarka meaningfulness (tests conclusion?)
- Hetvabhasa thoroughness (all 5 checked?)
- Nirnaya definitiveness
- Overall methodology quality

**Scoring Thresholds**:
```python
if total_score >= 0.85:  # 77/90 or higher
    return "ACCEPT"
elif total_score >= 0.70:  # 63-76/90
    return "MANUAL_REVIEW"
else:  # < 63/90
    return "REJECT"
```

**Expected Distribution**:
- 40-60% AUTO-ACCEPT
- 20-30% MANUAL_REVIEW
- 10-20% REJECT

**Cost**: ~$0.01-0.02 per evaluation → $5-10 for 500 examples

**Tier 3: Strategic Manual Review** (10-20% sample)

Not random sampling - strategic selection:
1. **Boundary cases** (scores 0.68-0.72): Calibrate LLM judge
2. **High-scoring validation** (scores >0.85): Ensure deserved
3. **Specific phase failures**: Examples weak in particular Nyaya component
4. **Problem type coverage**: Each type has manual validation

**Manual Review Time**: 50-75 examples → 15-20 hours

**Iterative Feedback Loop** (Weeks 11-16):

```python
def iterative_synthetic_generation(target_count=500, batch_size=100):
    """Generate with continuous quality improvement"""

    accepted = []
    generation_prompts = load_initial_prompts()

    while len(accepted) < target_count:
        # Generate batch
        raw_batch = generate_with_gpt4(batch_size, generation_prompts)

        # Tier 1: Structural filtering
        tier1_passed = [ex for ex in raw_batch if tier1_validation(ex)]

        # Tier 2: LLM judge
        tier2_results = [(ex, tier2_scoring(ex)) for ex in tier1_passed]
        tier2_accepted = [ex for ex, score in tier2_results if score == "ACCEPT"]

        # Tier 3: Manual review
        manual_queue = tier3_select_for_review(tier2_results)
        manually_validated = [ex for ex in manual_queue if manual_review(ex) == "ACCEPT"]

        # Analyze failures and update prompts
        failures = [ex for ex, score in tier2_results if score == "REJECT"]
        if failures:
            failure_patterns = analyze_common_errors(failures)
            generation_prompts = update_prompts(generation_prompts, failure_patterns)

        accepted.extend(tier2_accepted + manually_validated)

    return accepted[:target_count]
```

**Training** (Weeks 17-18):
- Retrain on expanded dataset (200-500 examples)
- Same hyperparameters as Stage 1
- Monitor for quality degradation from synthetic data

**Evaluation**:

**Custom Nyaya Metrics**:
- Pramana Validity Score (0-1): Are knowledge sources used correctly?
- Z3 Consistency Rate: Model conclusion matches Z3 on formal logic subset
- Hetvabhasa Detection: Precision/recall on adversarial fallacy examples

**Standard Benchmarks**:
- LogicBench (multi-step deduction)
- Full RuleTaker (complex rule sets)
- GSM8K subset (test domain transfer to math)

**Success Criteria**:
- ✅ 85-90% synthetic data quality (validated via sampling)
- ✅ Model maintains >90% format adherence
- ✅ Accuracy improves to 70-80% range
- ✅ Z3 consistency >85% on formal logic subset
- ✅ Demonstrates transfer to new problem domains

**Deliverables**:
- 200-500 validated examples (quality-controlled)
- Trained model on expanded dataset
- Benchmark results (LogicBench, ProntoQA, RuleTaker)
- **Hugging Face model release**: `pramana/nyaya-llama-8b-stage2`
- Published dataset on Hugging Face for community use

**Cost Breakdown**: ~$300-500 total
- GPT-4 evaluation: $50-100 (Tier 2 LLM judge for 500 examples @ $0.10 each)
- Synthetic generation: $200-300 (Claude/GPT-4 for generation @ $0.50 per example)
- Manual review: 20 hours (strategic sampling - your time)
- Compute: **$0** (DGX Spark)
- Tools: $0 (all open source)

**Community Impact**: Largest open Nyaya reasoning dataset, enables research on epistemological structure learning

---

### 5.3 Stage 3: Reinforcement Learning (OPTIONAL - 8-12 weeks)

**Strategic Decision Point**: Supervised fine-tuning (Stages 0-2) may be sufficient to prove the Pramana paradigm. Stage 3 adds refinement but at significant cost/complexity.

**Two Paths Forward**:

#### Path A: Deploy Stage 2 Model (RECOMMENDED for cost-effectiveness)

**When to choose**:
- Stage 2 achieves >85% format adherence
- 70-80% accuracy on benchmarks
- Clear epistemic advantages demonstrated
- Limited budget/time for RL experimentation

**Action**:
- Publish Stage 2 model as `pramana/nyaya-llama-8b-final`
- Write paper on Nyaya-structured fine-tuning approach
- Enable community to experiment with RL on top
- Focus on applications and case studies

**Total Project Cost**: ~$500-700 (Stages 0-2 only)

#### Path B: GRPO Enhancement (if budget/time available)

**When to choose**:
- Stage 2 shows promise but accuracy plateaued <70%
- Reward-based optimization might help quality
- DGX Spark available for 2-4 week continuous run
- Willing to invest time in RL infrastructure

**Goal**: Optimize for Nyaya-specific reward functions using Group Relative Policy Optimization **on DGX Spark**

**Cost-Effective GRPO Approach**:

Instead of $10-30K cloud compute, use:
1. **DGX Spark for training**: Free compute (4-8 A100s)
2. **Open-source PRM**: Train your own process reward model (don't use GPT-4 as judge)
3. **Smaller model**: Optimize 8B model, not 70B
4. **Shorter RL run**: 2 weeks max, not 4-8 weeks

**GRPO Configuration**:
```yaml
grpo:
  beta: 0.01                # KL penalty coefficient
  num_generations: 4        # Responses per prompt (>2 required)
  epsilon: 0.2              # PPO-style clipping value
  gamma: 1.0                # Discount factor
  lam: 0.95                 # GAE lambda

reward_functions:
  - name: nyaya_structure_completeness
    weight: 0.30
    description: "All six phases present and properly ordered"

  - name: logical_consistency
    weight: 0.25
    description: "Z3 verification passes on formal logic problems"

  - name: hetvabhasa_detection
    weight: 0.20
    description: "Correct identification of reasoning fallacies"

  - name: pramana_appropriateness
    weight: 0.15
    description: "Correct application of knowledge sources"

  - name: answer_correctness
    weight: 0.10
    description: "Final answer matches ground truth"
```

**Training Infrastructure** (DGX Spark):
- GPUs: 4-8 A100s on DGX Spark
- Duration: 2 weeks continuous training (not 4-8 weeks)
- Iterations: 150-200 GRPO iterations (not 300+)
- Learning rate: 5e-6 (very conservative for RL)

**Process Reward Model** (PRM) - Cost-Effective Approach:

**Recommended: Train Your Own PRM**
- Use Stage 2 examples with quality scores as training data
- Train small model (Llama 3.2 1B) to score Nyaya phases
- One-time training cost on DGX Spark (4 hours)
- Free inference during GRPO

**NOT Recommended: GPT-4 as judge**
- Would cost $5,000-10,000 for continuous RL scoring
- Unnecessary expense when you have labeled data
- Your PRM will be more calibrated to Nyaya specifics

**Composite Reward Function**:
```python
def compute_nyaya_reward(generated_solution, problem, ground_truth):
    """Calculate composite reward for GRPO"""

    rewards = {}

    # R1: Structural completeness (30%)
    structure = validate_nyaya_structure(generated_solution)
    rewards["structure"] = 0.30 * (
        1.0 if structure["has_all_phases"] else 0.0
    )

    # R2: Logical consistency via Z3 (25%)
    if is_formalizable(problem):
        z3_valid = verify_with_z3(generated_solution, problem)
        rewards["z3_consistency"] = 0.25 * (1.0 if z3_valid else -0.5)
    else:
        rewards["z3_consistency"] = 0.0

    # R3: Hetvabhasa detection (20%)
    fallacy_score = score_fallacy_detection(generated_solution)
    rewards["hetvabhasa"] = 0.20 * fallacy_score

    # R4: Pramana appropriateness (15%)
    pramana_score = score_pramana_usage(generated_solution, problem)
    rewards["pramana"] = 0.15 * pramana_score

    # R5: Answer correctness (10%)
    answer = extract_answer(generated_solution)
    rewards["answer"] = 0.10 * (1.0 if answer == ground_truth else 0.0)

    total_reward = sum(rewards.values())
    return total_reward, rewards
```

**Evaluation**:
- Reward metrics directly measure training objectives
- Reasoning efficiency: tokens per correct solution
- Competitive benchmarking: o1-preview, DeepSeek-R1, Claude 3.5
- Nyaya-specific benchmarks (competitive moat)

**Success Criteria**:
- ✅ High accuracy (80-90%+) with systematic reasoning
- ✅ Reward alignment validated (no gaming/hacking)
- ✅ Genuine epistemic improvements (not just performance)
- ✅ Model self-corrects via Tarka and Hetvabhasa

**Cost Breakdown (Path B - GRPO)**: ~$100-200
- PRM training data annotation: Manual effort (10-20 hours)
- Compute: **$0** (DGX Spark)
- Monitoring tools: $0 (open source - TensorBoard, W&B free tier)
- Optional: GPT-4 for PRM validation: $50-100
- **Total Stage 3: $100-200**

**Total Project Cost (All Stages 0-3)**: ~$600-900

**Success Criteria (Path B)**:
- ✅ Accuracy improves to 80-90%+ with GRPO
- ✅ Reward alignment validated (no gaming detected)
- ✅ Better than Stage 2 model on held-out tests
- ✅ Publishable improvement demonstrating RL benefit

**Deliverable**:
- **Hugging Face release**: `pramana/nyaya-llama-8b-grpo`
- Comparison paper: SFT vs GRPO for epistemological reasoning
- Community can reproduce RL approach

---

### 5.4 Community Deployment & Extensions (Post-Publication)

**Goal**: Enable community research and extensions, not solo production system

This stage is **community-driven**, not project-led. After publishing models and datasets on Hugging Face, researchers and practitioners can build:

**Potential Community Extensions**:

1. **Multi-Agent Debate Protocols**:
   - Vada (cooperative), Jalpa (adversarial), Vitanda (critical) debate systems
   - Use published Pramana models as debating agents
   - Research: Does structured debate reduce hallucination?

2. **Neuro-Symbolic Integration**:
   - Runtime Z3 verification for formal logic subset
   - Autoformalization from Nyaya traces to SMT-LIB
   - "Proof of Thought" pipeline for mathematical certainty

3. **Domain Extensions**:
   - Apply Nyaya to causal reasoning (beyond formal logic)
   - Software debugging with Hetvabhasa detection
   - Mathematical theorem proving with structured justification
   - Legal reasoning with precedent-based Shabda

4. **Optimization & Deployment**:
   - vLLM serving optimization
   - Quantization to 4-bit for edge deployment
   - Fast path for simple problems, full Nyaya for complex
   - Production APIs and applications

5. **Cross-Lingual Nyaya**:
   - Translate approach to other languages
   - Original Sanskrit terminology integration
   - Multilingual reasoning datasets

**Your Role Post-Publication**:
- Maintain model cards and documentation
- Review community PRs to dataset/evaluation
- Publish updates if you create new training data
- Cite and promote community extensions
- Not responsible for production deployment

**Success Metrics (Community Adoption)**:
- Downloads on Hugging Face (target: 1,000+ in first 6 months)
- Citations in academic papers
- Derivative models and datasets
- GitHub stars/forks on repository
- Community contributions (PRs, issues, discussions)

**Cost**: $0 (community-driven innovation)

---

## 6. Risk Mitigation Strategy

### Risk 1: Syntactic Mimicry Without Semantic Reasoning

**Symptom**: Model generates perfect format but logically incoherent content

**Detection**:
- Z3 verification fails on formal logic problems
- Human evaluation shows invalid reasoning steps
- Tarka sections don't actually test conclusions
- Udaharana provides examples without universal rules

**Mitigation**:
- Stage 2 LLM judge with explicit semantic rubrics
- Extensive human evaluation (Tier 3 manual review)
- Z3 verification on formal logic subset
- Adversarial test cases with intentional traps

**Pivot Strategy**:
- Increase manual seed examples (more gold standard demonstrations)
- Improve Udaharana teaching (emphasize "Wherever X, there is Y" requirement)
- Add negative examples showing what NOT to do
- Simplify problem types to isolate learning

### Risk 2: Domain Overfitting

**Symptom**: Works brilliantly on CSP but fails on Boolean SAT or other types

**Detection**:
- Performance drops >30% on new problem types
- Model applies CSP-specific reasoning to all problems
- Benchmark performance varies wildly across domains

**Mitigation**:
- Test on diverse problems early in Stage 1
- Expand problem type diversity in Stage 2
- Monitor per-type performance metrics
- Ensure training data balanced across types

**Pivot Strategy**:
- Create more examples in struggling problem types
- Add intermediate complexity levels
- Teach domain-agnostic Nyaya principles explicitly
- Reduce concentration on any single problem type

### Risk 3: Synthetic Data Poisoning

**Symptom**: Scaled generation introduces subtle systematic errors

**Detection**:
- Three-tier quality control catches patterns
- Model performance regresses with synthetic data
- Human review identifies recurring mistakes
- Z3 verification fails systematically on certain patterns

**Mitigation**:
- GPT-4 judge with explicit Nyaya rubric
- Iterative prompt refinement based on failure analysis
- Statistical sampling validation
- Z3 auto-verification where applicable

**Pivot Strategy**:
- Fall back to more manual review (increase Tier 3 percentage)
- Reduce synthetic ratio, increase manual examples
- Filter more aggressively (accept only >90% scores)
- Use multiple generation models to avoid systematic biases

### Risk 4: Reasoning Overhead Unacceptable

**Symptom**: 6-phase structure produces 2000 tokens vs 200 for GPT-4

**Detection**:
- Token/problem tracking shows 5-10x overhead
- Inference time >5s per problem
- Cost per query prohibitively high
- Users complain about verbosity

**Mitigation**:
- Monitor efficiency metrics from Stage 0
- Measure overhead vs. accuracy tradeoff
- Consider abbreviated forms for simple cases
- Optimize vLLM serving in Stage 4

**Pivot Strategy**:
- Design "fast path" for trivial problems (skip full Nyaya)
- Teach model to self-assess problem complexity
- Create condensed format option
- Accept overhead as feature not bug (interpretability value)

### Risk 5: Burnout / Time Constraints

**Symptom**: Loss of momentum during manual seed creation (e.g., stuck at Problem 15 of 50)

**Detection**:
- Quality degradation in later examples
- Missed deadlines
- Loss of rigor in Nyaya methodology
- Frustration with repetitive work

**Mitigation**:
- Staged approach provides value at each checkpoint
- Can stop after Stage 1 with useful artifact
- Break work into manageable chunks (12-13 examples/week)
- Vary problem types to maintain interest

**Pivot Strategy**:
- Scale down to 25 examples instead of 50 in Stage 1
- Slower pace with quality focus (better 30 excellent than 50 mediocre)
- Enlist collaborators if available
- Take breaks between stages

---

## 7. Evaluation Framework

**Philosophy**: Staged benchmarking that evolves with project maturity

### 7.1 Stage 0: Structural Validity (Format-First)

**Primary Metric**: Nyaya Phase Completeness (Binary)
```python
{
    "has_all_phases": bool,
    "correct_order": bool,
    "phase_components": {
        "samshaya": {"doubt_type_identified": bool},
        "pramana": {
            "pratyaksha": bool,
            "anumana": bool,
            "upamana": bool,
            "shabda": bool
        },
        "pancha_avayava": {
            "pratijna": bool,
            "hetu": bool,
            "udaharana": bool,
            "upanaya": bool,
            "nigamana": bool,
            "universal_rule_present": bool  # "Wherever X, there is Y"
        },
        "tarka": {"counterfactual_present": bool},
        "hetvabhasa": {"all_five_checked": bool},
        "nirnaya": {"conclusion_stated": bool}
    }
}
```

**Target**: 80%+ completeness on held-out example

**Secondary Metric**: Phase Content Appropriateness (Manual)
- Does Samshaya identify correct doubt type?
- Does Pratyaksha contain ONLY observables?
- Does Udaharana have universal rule?
- Does Tarka actually test conclusion?
- Are all Hetvabhasa checked meaningfully?

**Why Accuracy is Secondary**: With 5 examples, model overfits. Key question: *When it makes mistakes, does it make them through Nyaya structure or by abandoning it?*

### 7.2 Stage 1: Structure + Correctness Correlation

**Format Adherence**: Target 90%+

**Accuracy by Phase Quality** (Critical Innovation):
```python
correlation_metrics = {
    "all_phases_present": {
        "count": X,
        "accuracy": Y%  # Should be highest
    },
    "missing_tarka": {
        "count": X,
        "accuracy": Y%  # Should be lower
    },
    "incomplete_pramana": {
        "count": X,
        "accuracy": Y%  # Should be lower
    },
    "weak_udaharana": {
        "count": X,
        "accuracy": Y%  # Should be lower
    }
}
```

**Critical Validation**: If accuracy is independent of structure quality, the Nyaya approach has failed. Structure must help reasoning, not just decorate it.

**Benchmark Introduction**:
- ProntoQA (logical inference)
- Simple RuleTaker (rule-based reasoning)
- Not expecting to beat baselines - measuring transfer

### 7.3 Stage 2: Custom Nyaya Scoring

**Pramana Validity Scoring** (0-1 scale):
```python
def score_pramana_application(trace):
    scores = {
        "pratyaksha_valid": 0.25,  # Only observable facts?
        "anumana_valid": 0.25,     # Actual logical inferences?
        "upamana_relevant": 0.25,  # Appropriate analogies?
        "shabda_correct": 0.25     # Valid principles?
    }
    return sum(validate_each(trace, scores))
```

**Z3 Consistency Rate**:
- Percentage of formal logic problems where model matches Z3
- Only applicable to CSP/Boolean SAT subset (~30%)
- Target: 90%+ consistency

**Hetvabhasa Detection Accuracy**:
- Create adversarial examples with intentional fallacies
- Precision/recall on detecting which type
- Target: 70%+ F1 score

**Standard Benchmarks**:
- LogicBench (multi-step deduction)
- Full RuleTaker (complex rules)
- GSM8K subset (domain transfer to math)

### 7.4 Stage 3: Reward-Aligned Evaluation

**Composite Score** (mirrors GRPO weights):
```python
evaluation_score = (
    0.30 * structure_completeness +
    0.25 * z3_consistency +
    0.20 * hetvabhasa_accuracy +
    0.15 * pramana_appropriateness +
    0.10 * answer_correctness
)
```

**Reasoning Efficiency**:
- Average tokens per correct solution
- Nyaya overhead vs standard CoT
- Does structure reduce trial-and-error iterations?

**Reward Hacking Detection**:
- Monitor for gaming the reward function
- Ensure improvements are genuine, not exploiting loopholes
- Test on out-of-distribution problems

### 7.5 Stage 4: Competitive Benchmarking

**Frontier Model Comparison**:
- o1-preview on reasoning benchmarks
- DeepSeek-R1 on mathematical reasoning
- Claude 3.5 on logical puzzles

**Nyaya-Specific Benchmarks** (Competitive Moat):
- Nyaya Fallacy Detection Suite (custom)
- Pramana Source Validation (knowledge source correctness)
- Multi-step Inference with Explicit Warrants (show your work)

**Interpretability Metrics**:
- Can users trace reasoning steps?
- Are failure modes identifiable from traces?
- Does structured output help debugging?
- Survey: Do users trust reasoning more than black-box?

---

## 8. Project Structure & Development Workflow

### 8.1 Directory Structure

```
pramana-project/
├── data/
│   ├── seed_examples/
│   │   ├── stage_zero/           # 5 manual examples
│   │   ├── stage_one/            # 50 manual examples
│   │   │   ├── constraint_satisfaction/
│   │   │   ├── boolean_sat/
│   │   │   └── multi_step_deduction/
│   │   └── stage_two_synthetic/  # 200-500 synthetic
│   │       ├── batch_001/
│   │       ├── batch_002/
│   │       └── tier1_passed/
│   ├── validation/
│   │   ├── held_out/             # Never-seen test cases
│   │   └── benchmarks/           # LogicBench, ProntoQA, etc.
│   └── evaluation/
│       └── adversarial/          # Hetvabhasa test cases
│
├── models/
│   ├── checkpoints/
│   │   ├── stage_zero/
│   │   ├── stage_one/
│   │   ├── stage_two/
│   │   └── stage_three_grpo/
│   └── final/                    # Production models
│
├── scripts/
│   ├── data/
│   │   ├── parse_markdown_examples.py
│   │   ├── validate_nyaya_structure.py
│   │   └── synthetic_generation.py
│   ├── training/
│   │   ├── stage_zero_finetune.py
│   │   ├── stage_one_finetune.py
│   │   ├── stage_two_finetune.py
│   │   └── stage_three_grpo.py
│   ├── evaluation/
│   │   ├── tier1_structural_validation.py
│   │   ├── tier2_llm_judge.py
│   │   ├── tier3_manual_review_ui.py
│   │   ├── z3_verification.py
│   │   └── benchmark_runner.py
│   └── validation/
│       ├── z3_autoformalize.py
│       └── neuro_symbolic_pipeline.py
│
├── configs/
│   ├── stage_zero_config.yaml
│   ├── stage_one_config.yaml
│   ├── stage_two_config.yaml
│   └── grpo_stage_three_config.yaml
│
├── results/
│   ├── experiments/              # W&B logs, tensorboard
│   ├── evaluations/              # Benchmark results
│   └── analysis/                 # Error analysis, ablations
│
├── docs/
│   ├── plans/                    # Design documents
│   │   └── spec.md               # This file
│   ├── nyaya_glossary.md         # Terms and concepts
│   ├── eval_strategy.txt         # Evaluation methodology
│   ├── seed_example_type.txt     # Example guidelines
│   └── synth_data.txt            # Synthetic generation strategy
│
├── Dockerfile
├── pramana_docker_setup.sh
├── run_pramana_container.sh
├── run_jupyter.sh
├── CLAUDE.md
└── README.md
```

### 8.2 Development Workflow

**Daily Development Cycle**:
```bash
# 1. Launch container
cd ~/pramana-project
./run_pramana_container.sh

# 2. Start monitoring (optional)
tensorboard --logdir results/ --port 6006 &

# 3. Work on task (example creation, training, eval)
vim data/seed_examples/stage_zero/problem_03.md

# 4. Validate structure
python scripts/data/validate_nyaya_structure.py \
  --input data/seed_examples/stage_zero/problem_03.md

# 5. Commit progress
git add data/seed_examples/stage_zero/problem_03.md
git commit -m "Add 4-variable CSP example with spatial constraints"
```

**Training Workflow**:
```bash
# Validate all data
python scripts/data/validate_nyaya_structure.py \
  --input data/seed_examples/stage_zero/ \
  --recursive

# Run training
python scripts/training/stage_zero_finetune.py \
  --config configs/stage_zero_config.yaml \
  --wandb_project pramana-stage-zero \
  --experiment_name "attempt_02_higher_lora_rank"

# Monitor training
# Access W&B dashboard or TensorBoard at localhost:6006

# Evaluate checkpoint
python scripts/evaluation/stage_zero_eval.py \
  --checkpoint models/checkpoints/stage_zero/final \
  --validation data/validation/held_out/zebra.md \
  --output results/evaluations/stage_zero_final.json
```

**Quality Control Workflow** (Stage 2):
```bash
# Generate synthetic batch
python scripts/data/synthetic_generation.py \
  --num_examples 100 \
  --generation_model gpt-4-turbo-preview \
  --seed_examples data/seed_examples/stage_one/ \
  --output data/stage_two_synthetic/batch_001/

# Tier 1: Automated filtering
python scripts/evaluation/tier1_structural_validation.py \
  --input data/stage_two_synthetic/batch_001/ \
  --output data/stage_two_synthetic/batch_001/tier1_passed/

# Tier 2: LLM judge
python scripts/evaluation/tier2_llm_judge.py \
  --input data/stage_two_synthetic/batch_001/tier1_passed/ \
  --judge_model gpt-4-turbo-preview \
  --output data/stage_two_synthetic/batch_001/tier2_scored/

# Tier 3: Manual review UI
python scripts/evaluation/tier3_manual_review_ui.py \
  --input data/stage_two_synthetic/batch_001/tier2_scored/manual_review_queue/ \
  --output data/stage_two_synthetic/batch_001/accepted/

# Analyze failures and update prompts
python scripts/data/analyze_generation_failures.py \
  --rejected data/stage_two_synthetic/batch_001/tier2_scored/rejected/ \
  --output docs/generation_improvements_v2.md
```

---

## 9. Success Gates & Decision Points

### 9.1 Stage 0 → Stage 1 Gate

**GO Criteria** (all must pass):
- ✅ Model achieves 80%+ structural completeness on held-out Zebra puzzle
- ✅ At least 3 valid Pancha Avayava syllogisms present per solution
- ✅ Zero instances of completely abandoning structure mid-solution
- ✅ Correct answer on 60%+ of generation attempts (even if overfitted)
- ✅ Manual evaluation confirms phases contain appropriate content (not just template-filling)

**NO-GO Criteria** (any one fails the stage):
- ❌ <50% structural adherence on held-out example
- ❌ Model generates generic CoT instead of Nyaya structure
- ❌ Correct answers achieved without using Nyaya methodology
- ❌ Complete inability to learn any aspect of structure after 10 epochs

**Pivot Options**:
- **Partial success (60-79% structure)**: Try different base model (Qwen vs DeepSeek), increase examples to 8-10
- **Format issues**: Simplify to 4 phases (Pramana, Avayava, Tarka, Nirnaya)
- **Learning issues**: Increase LoRA rank to 128, train for 15 epochs
- **Wrong base model**: Switch between DeepSeek and Qwen

**Decision Timeline**: End of Week 2

---

### 9.2 Stage 1 → Stage 2 Gate

**GO Criteria** (all must pass):
- ✅ 90%+ format adherence on held-out problems (10+ examples)
- ✅ 60-70% accuracy range achieved (establishing baseline)
- ✅ Positive correlation demonstrated: full structure → higher accuracy
- ✅ Generalizes to ≥2 problem types (e.g., CSP + Boolean SAT both work)
- ✅ No catastrophic forgetting (still performs on Stage 0 validation)

**NO-GO Criteria**:
- ❌ Accuracy independent of structural quality (correlation near zero)
- ❌ Works only on CSP, fails catastrophically on Boolean SAT (<30% structure)
- ❌ Format adherence degrades below 80%
- ❌ Model worse than Stage 0 (regression)

**Pivot Options**:
- **Domain overfitting**: Create 15-20 more examples in struggling problem type before scaling
- **Structure-accuracy decoupling**: Revisit Udaharana teaching, emphasize universal rules
- **Quality issues**: Improve seed example quality, add negative examples
- **Scaling too fast**: Stay at 50 examples, perfect training before synthetic scaling

**Decision Timeline**: End of Week 10

---

### 9.3 Stage 2 → Stage 3 Gate

**GO Criteria**:
- ✅ 85-90% synthetic data quality validated through three-tier process
- ✅ Model maintains >90% format adherence with expanded dataset
- ✅ Accuracy improves to 70-80% range (not just maintaining)
- ✅ Z3 consistency >85% on formal logic subset
- ✅ Demonstrates transfer to new domains (passes benchmark tests)
- ✅ No evidence of synthetic data poisoning (quality stable across batches)

**NO-GO Criteria**:
- ❌ Synthetic data quality <75% despite three-tier filtering
- ❌ Performance regresses with synthetic data vs. manual-only
- ❌ No improvement over Stage 1 metrics (plateaued)
- ❌ Systematic errors in synthetic data that can't be fixed

**Pivot Options**:
- **Quality issues**: Reduce synthetic ratio (70% synthetic, 30% manual instead of 90/10)
- **Prompt problems**: Improve GPT-4 judge rubric, stricter thresholds (>0.90 to accept)
- **Cost concerns**: Scale to 300 examples instead of 500
- **Skip RL**: Deploy Stage 2 model without GRPO, iterate on SFT approach

**Decision Timeline**: End of Week 18

---

### 9.4 Stage 3 → Stage 4 Gate

**GO Criteria**:
- ✅ High accuracy (80-90%+) with systematic reasoning
- ✅ Reward alignment validated (genuine improvements, no reward hacking detected)
- ✅ Competitive or near-competitive with frontier models on select benchmarks
- ✅ Clear epistemic advantages: interpretability, self-correction capability
- ✅ Cost acceptable for deployment (<$0.05/query at scale)

**NO-GO Criteria**:
- ❌ GRPO doesn't improve over Stage 2 SFT model
- ❌ Reward hacking detected (gaming metrics without genuine reasoning improvement)
- ❌ Prohibitively expensive to train (>$50k) or serve (>$0.10/query)
- ❌ No clear advantage over frontier models (performance parity but no interpretability benefit)

**Pivot Options**:
- **RL doesn't help**: Skip Stage 4, deploy Stage 2 model as-is
- **Reward function issues**: Adjust weights, add new reward components
- **Cost reduction**: Use smaller model (7B instead of 14B), optimize serving
- **Hybrid approach**: GRPO for hard problems, SFT for easy ones

**Decision Timeline**: End of Week 30-32

---

## 10. Glossary of Nyaya Terms

**Samshaya**: Doubt or uncertainty requiring systematic investigation. Five types: Samana Dharma Upapatti, Aneka Dharma Upapatti, Vipratipatti, Upalabdhi Avyavastha, Anupalabdhi Avyavastha.

**Pramana**: Valid means of knowledge. Four types recognized in Nyaya.

**Pratyaksha**: Direct perception/observation through senses.

**Anumana**: Logical inference. Three types: Purvavat (cause→effect), Sheshavat (effect→cause), Samanyatodrishta (general correlation).

**Upamana**: Knowledge through comparison or analogy to known cases.

**Shabda**: Authoritative testimony or established logical principles.

**Pancha Avayava**: Five-member syllogism, the core deductive structure.

**Pratijna**: Proposition or thesis to be proved.

**Hetu**: Reason or evidence supporting the thesis.

**Udaharana**: Universal rule plus concrete example (dṛṣṭānta). Must contain "Wherever X, there is Y" structure.

**Upanaya**: Application of universal rule to the specific case at hand.

**Nigamana**: Conclusion that follows from the above four components.

**Vyapti**: Universal concomitance or invariable relation between two phenomena.

**Tarka**: Hypothetical or counterfactual reasoning, typically reductio ad absurdum.

**Hetvabhasa**: Logical fallacies or pseudo-reasons. Five types: Savyabhichara, Viruddha, Prakaranasama, Sadhyasama, Kalaatita.

**Nirnaya**: Definitive ascertainment or conclusion. Distinguishes established knowledge from hypothesis.

**Vada**: Proper philosophical debate for collaborative truth-seeking.

**Jalpa**: Sophisticated debate aimed at victory through valid argumentation.

**Vitanda**: Critical debate focused on finding flaws without proposing alternatives.

---

## 11. Timeline & Resource Summary

### 11.1 Revised Timeline (Focused on DGX Spark + Hugging Face)

**Path A: SFT Only (RECOMMENDED)**

**Stage 0: Weeks 1-2**
- 5 seed examples, proof of concept
- Cost: ~$100

**Stage 1: Weeks 3-10**
- 50 seed examples, minimum viable reasoner
- **Hugging Face Release 1**: `pramana/nyaya-llama-8b-stage1`
- Cost: ~$200

**Stage 2: Weeks 11-18**
- 200-500 synthetic examples, quality-controlled scaling
- **Hugging Face Release 2**: `pramana/nyaya-llama-8b-stage2`
- **Dataset Release**: Training data for community use
- Cost: ~$300-500

**Total Timeline**: 18 weeks (4.5 months)
**Total Cost**: ~$600-800

---

**Path B: SFT + GRPO (OPTIONAL)**

Add Stage 3 if needed:

**Stage 3: Weeks 19-22**
- GRPO reinforcement learning
- **Hugging Face Release 3**: `pramana/nyaya-llama-8b-grpo`
- Cost: ~$100-200

**Total Timeline**: 22 weeks (5.5 months)
**Total Cost**: ~$700-1000

---

**Community Extensions (Post-Publication)**
- Timeline: Indefinite, community-driven
- Cost: $0 (not your responsibility)

### 11.2 Resource Requirements (Path A - Recommended)

**Human Effort**:
- Stage 0: 15-20 hours (example creation + evaluation)
- Stage 1: 60-80 hours (50 examples + training)
- Stage 2: 20-30 hours (synthetic data review)
- **Total: ~95-130 hours over 4.5 months** (~5-6 hours/week)

**Compute Resources** (All on DGX Spark - Free):
- Stage 0: 4-6 GPU-hours (single A100)
- Stage 1: 40-60 GPU-hours (single A100)
- Stage 2: 50-80 GPU-hours (single A100)
- **Total: ~94-146 GPU-hours** (all DGX Spark, $0 cost)

**Financial Budget** (Path A):
- Stage 0: ~$100 (GPT-4 for quality checks)
- Stage 1: ~$200 (GPT-4 evaluation/scoring)
- Stage 2: ~$300-500 (synthetic generation + evaluation)
- **Total: $600-800** (vs original $12,600-36,100!)

**Key Cost Savings**:
- ✅ DGX Spark compute: $0 (vs $15,000+ cloud GPU)
- ✅ No production deployment: $0 (community handles)
- ✅ Stage 3 optional: Save $10-30K if SFT sufficient
- ✅ Focus on proof-of-concept, not production system

### 11.3 Key Milestones (Path A - Recommended)

**Month 1 (Weeks 1-4)**:
- ✓ Stage 0 complete (Week 2)
- ✓ 15 Stage 1 examples created (Week 4)
- **Deliverable**: Proof that Nyaya structure is learnable
- Decision: GO/NO-GO for full Stage 1

**Month 2 (Weeks 5-8)**:
- ✓ 40 Stage 1 examples complete
- ✓ Model training initiated
- **Deliverable**: Progress on seed dataset

**Month 3 (Weeks 9-12)**:
- ✓ 50 Stage 1 examples complete
- ✓ Stage 1 model trained and evaluated
- ✓ Structure-accuracy correlation validated
- **🚀 HUGGING FACE RELEASE 1**: `pramana/nyaya-llama-8b-stage1`
- Decision: GO/NO-GO for Stage 2

**Month 4 (Weeks 13-16)**:
- ✓ Three-tier quality pipeline operational
- ✓ 200 synthetic examples generated
- ✓ Model trained on expanded dataset
- **Deliverable**: Scaled dataset with quality control

**Month 5 (Weeks 17-18)**:
- ✓ Stage 2 complete (300-500 examples)
- ✓ Benchmark evaluation on LogicBench, ProntoQA
- **🚀 HUGGING FACE RELEASE 2**: `pramana/nyaya-llama-8b-stage2`
- **📊 DATASET RELEASE**: All training data published
- **📝 PAPER DRAFT**: Write-up of Nyaya fine-tuning approach
- **Decision**: Deploy Stage 2 OR proceed to Stage 3 (GRPO)

**Optional Month 6 (Weeks 19-22)** - If Path B chosen:
- ✓ GRPO training on DGX Spark
- ✓ PRM trained and deployed
- **🚀 HUGGING FACE RELEASE 3**: `pramana/nyaya-llama-8b-grpo`
- **📝 COMPARISON PAPER**: SFT vs GRPO for epistemological reasoning

**Post-Publication**:
- Community extensions and applications
- Monitor Hugging Face downloads and citations
- Support community PRs and issues

### 11.4 Hardware Contingency Plan

**Primary**: DGX Spark (NVIDIA platform)
- Reliable infrastructure
- Free compute for research
- 4-8 A100 GPUs available

**Backup** (if DGX Spark unavailable):

**Option 1: Cloud GPU (Lambda Labs / RunPod)**
- Cost: ~$1.50-2.00/hour per A100
- Stage 0: ~$10 (6 hours)
- Stage 1: ~$120 (60 hours)
- Stage 2: ~$160 (80 hours)
- **Total backup cost**: ~$300 vs $0 on DGX

**Option 2: Consumer GPU (RTX 4090)**
- Works for Stage 0-1 (8B model fits in 24GB with 4-bit)
- Stage 2-3 might require model parallelism or smaller models
- Free if you have access to hardware

**Data Backup Strategy**:
- **Git**: All examples, configs, scripts
- **HuggingFace**: Model checkpoints (free up to 100GB)
- **External HDD**: Local backup of all artifacts
- **Sync frequency**: Daily for active training, weekly for stable checkpoints

**Disaster Recovery**:
- Training crash: Resume from last checkpoint (saved every 100 steps)
- Data corruption: Restore from Git + HuggingFace
- Hardware failure: Switch to backup GPU within 24 hours
- Complete loss: All artifacts recoverable from backups

---

## 12. References & Resources

### Key Papers
1. Nyaya Sutras - Gautama Maharishi (~500 BCE)
2. Tattvacintāmaṇi - Gaṅgeśa Upādhyāya (1325 CE)
3. DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning (2025)
4. Apple ML Research: "GSM-Symbolic: Understanding the Limitations of Mathematical Reasoning in LLMs" (Oct 2024)
5. Group Relative Policy Optimization (GRPO) Paper
6. Process Reward Models for RLHF Research
7. Unsloth: Efficient Fine-tuning of Large Language Models

### Technical Documentation
- Unsloth Documentation: https://docs.unsloth.ai/
- Z3 Theorem Prover: https://github.com/Z3Prover/z3
- vLLM Documentation: https://docs.vllm.ai/
- Weights & Biases: https://docs.wandb.ai/

### Benchmark Datasets
- LogicBench: https://github.com/Mihir3009/LogicBench
- ProntoQA: Ontological reasoning dataset
- RuleTaker: Rule-based logical reasoning
- GSM8K: Mathematical word problems

### Project Documentation
- `docs/nyaya_glossary.md`: Detailed Nyaya terminology
- `docs/eval_strategy.txt`: Comprehensive evaluation methodology
- `docs/seed_example_type.txt`: Guidelines for example creation
- `docs/synth_data.txt`: Three-tier synthetic data pipeline
- `CLAUDE.md`: Project instructions for AI assistants
- `README.md`: Quick start and project overview

---

## 13. Ethical Considerations & Limitations

**Epistemic Responsibilities**:
- Model designed for **logical/mathematical domains** (CSP, SAT, deduction)
- **NOT validated** for moral reasoning, value judgments, or policy decisions
- Nyaya structure provides interpretability, but **not correctness guarantee**
- Structured output can be wrong - always verify high-stakes decisions

**Potential Misuse Vectors**:
1. **Authoritative-Sounding Falsehoods**: Generating logically flawed reasoning that looks rigorous due to formal structure
2. **Obscuring Weak Arguments**: Using formal Nyaya structure to make weak claims appear stronger
3. **Over-Trust in Structure**: Assuming structured output is correct without verification (structure ≠ correctness)
4. **Hallucinated Sources**: Shabda (testimony) citations that don't exist

**Mitigation Strategies**:
- **Clear Documentation**: Model card explicitly states "Pramana is logic tool, not truth oracle"
- **Watermark Outputs**: Include "[Reasoning via Nyaya methodology - verify conclusions]"
- **Encourage Verification**: Documentation emphasizes human review for high-stakes decisions
- **Public Model AUP**: Acceptable Use Policy prohibits misuse for misinformation

**Acknowledged Limitations**:
- Training limited to **formal logic domains** (constraint satisfaction, propositional logic)
- May **not transfer** to fuzzy/probabilistic reasoning, moral philosophy, or aesthetic judgment
- **Cultural assumptions**: Nyaya is Indian logic tradition, not universal epistemology
- **English-language only**: Nyaya Sanskrit terms translated, may lose nuance
- **Format overhead**: 3-6x more verbose than standard CoT (efficiency cost)

**Transparency Obligations**:
- All training data published on Hugging Face
- Model limitations clearly documented in model card
- Evaluation metrics include failure modes
- Community can audit and critique approach

**Research Ethics**:
- No human subjects (synthetic data generation)
- No sensitive/private data in training
- Open science: reproducible, auditable, extensible
- Credit to Nyaya Darshan tradition and scholars

---

## 14. Reproducibility Checklist

To ensure this work is fully reproducible by the research community:

**Code** (Git Repository):
- [ ] All scripts version-controlled in public Git repo
- [ ] Docker environment fully specified (Dockerfile + setup script)
- [ ] Dependency versions pinned (`requirements.txt` with `==` notation)
- [ ] Random seeds set and documented for all training runs
- [ ] Training scripts include hyperparameter logging

**Data** (Hugging Face Datasets):
- [ ] All seed examples committed to Git (`.md` files)
- [ ] Synthetic generation prompts saved and versioned
- [ ] Train/validation/test splits documented with split seeds
- [ ] Data preprocessing code reproducible (parsing scripts)
- [ ] `.dataversion` files track data quality and changes
- [ ] Dataset published on Hugging Face with DOI

**Training** (Experiment Tracking):
- [ ] Hyperparameters logged to Weights & Biases
- [ ] Checkpoints saved with metadata (epoch, loss, timestamp, git commit)
- [ ] Training curves exportable (TensorBoard logs)
- [ ] GPU configuration documented (DGX Spark A100 setup)
- [ ] Environment variables documented (CUDA version, etc.)

**Evaluation** (Reproducible Metrics):
- [ ] Evaluation scripts deterministic (temperature=0 for scoring)
- [ ] Manual review decisions logged with justification in separate file
- [ ] Benchmark versions specified (LogicBench commit hash, dataset version)
- [ ] Metrics computation code unit-tested for correctness
- [ ] Ablation study configurations saved

**Publication Artifacts** (for Paper + Hugging Face):
- [ ] 5-10 best examples from each stage (gold standard demonstration)
- [ ] Model checkpoints on HuggingFace (Stage 1, 2, optionally 3)
- [ ] Evaluation harness open-sourced (can reproduce benchmark results)
- [ ] Supplementary materials with full reasoning traces
- [ ] Model cards with detailed usage instructions
- [ ] Citation information (BibTeX)

**Hardware Specifications** (for Replication):
- [ ] DGX Spark configuration documented
- [ ] Backup hardware options tested and documented
- [ ] Cloud GPU costs estimated for community replication
- [ ] Consumer GPU viability tested (RTX 4090 for Stage 0-1)

**Community Support** (Post-Publication):
- [ ] GitHub issues enabled for questions and bug reports
- [ ] Model card includes contact information
- [ ] Hugging Face discussions enabled
- [ ] README includes "How to Replicate" section
- [ ] Known issues and limitations documented

---

**END OF SPECIFICATION**

**Version**: 1.1 (Revised)
**Last Updated**: 2025-01-30
**Changes from v1.0**:
- Refocused on DGX Spark (free compute), removed expensive cloud options
- Made Stage 3 (GRPO) optional, reduced total cost from $12-36K to ~$600-900
- Emphasized Hugging Face sharing and community building (not solo production)
- Added high-priority review items: overfitting justification, shortcut detection, failure recovery
- Added negative examples, data versioning, ethical considerations
- Shortened timeline from 32 weeks to 18-22 weeks

**Status**: Ready for Stage 0 Implementation
**Next Action**: Create first seed example (Problem 1: 3-variable CSP with direct constraints)



