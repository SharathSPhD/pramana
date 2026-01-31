# Stage 0 Corrective Action Plan

**Date**: 2026-01-31  
**Status**: Draft - Ready for Implementation  
**Priority**: CRITICAL - Stage 0 Failed Format Learning

---

## Executive Summary

Stage 0 training completed successfully but **failed to achieve format learning** (0% format adherence). The model produces generic chain-of-thought reasoning instead of the required 6-phase Nyaya structure. This document provides a detailed, actionable plan to correct the root causes and achieve Stage 0 success criteria.

**Critical Findings**:
- Format adherence: **0%** (0/2 test examples)
- Model learned *content* (correct answers) but not *structure* (Nyaya format)
- Root causes: Insufficient data (5 examples), low LoRA capacity (rank 32), missing format enforcement

**Target**: Achieve **>80% format adherence** with corrected implementation.

---

## 1. Problem Statement

### 1.1 What Failed

**Training Outcome**: Model training completed successfully (50 steps, 25 epochs, final loss 0.9898). LoRA adapters were saved to `models/stage_0/`.

**Evaluation Outcome**: Post-training evaluation on held-out examples (`pramana-003`, `pramana-005`) revealed **complete format learning failure**:
- **Format adherence**: 0% (0/2 examples)
- **Parseable outputs**: 0% (0/2 examples)
- **Answer correctness**: 100% (answers are correct, but format is wrong)

**Model Behavior**: The fine-tuned model produces generic chain-of-thought reasoning identical to the base model, with no evidence of Nyaya structure learning. Example output:

```
We need to reason through this problem using Nyaya logic rules...

**Step 1**: Given that P is true, we can infer that Q is true...
**Step 2**: Given that Q is true, we can infer that R is true...
```

**Expected Behavior**: Structured 6-phase Nyaya output:

```markdown
## Samshaya (Doubt Analysis)
**Doubt Type**: Viparyaya Samshaya...

## Pramana (Sources of Knowledge)
### Pratyaksha (Direct Perception)
...

## Pancha Avayava (5-Member Syllogism)
### Syllogism 1: Establishing Q from P
**Pratijna (Thesis)**: Q is true
**Hetu (Reason)**: P is true and P → Q
...
```

### 1.2 Why It Failed

**Root Cause Analysis** (from `docs/implementation_report.md`):

1. **Insufficient Training Data** (CRITICAL)
   - **Current**: 5 examples (1 per problem type)
   - **Problem**: Model cannot generalize format from 5 examples
   - **Evidence**: Model memorized content but not structure
   - **Impact**: HIGH - Format learning requires repeated exposure to structure

2. **LoRA Capacity Too Low** (CRITICAL)
   - **Current**: Rank 32, Alpha 32 (48.6M trainable params, ~1.49% of model)
   - **Problem**: Insufficient capacity to override base model's CoT format
   - **Evidence**: Model reverted to base instruction-following behavior
   - **Impact**: HIGH - Complex format learning requires higher capacity

3. **Missing Format Enforcement** (CRITICAL)
   - **Current**: Generic prompt `### Problem: ... ### Nyaya Reasoning:`
   - **Problem**: No explicit format requirements in system/user prompts
   - **Evidence**: Model treats "Nyaya Reasoning" as semantic label, not structural requirement
   - **Impact**: HIGH - Model needs explicit format instructions

4. **Sequence Length Too Short** (MEDIUM)
   - **Current**: 2048 tokens
   - **Problem**: May truncate long Nyaya reasoning traces
   - **Impact**: MEDIUM - Full traces require 3000-4000 tokens

5. **Overfitting Risk** (MEDIUM)
   - **Current**: 25 epochs on 5 examples (125 training steps per example)
   - **Problem**: Extreme overfitting, no validation split
   - **Evidence**: Final loss (0.9898) higher than minimum (0.8084)
   - **Impact**: MEDIUM - May prevent generalization

6. **Base Model Interference** (MEDIUM)
   - **Current**: Llama-3.2-3B-Instruct (strong CoT priors)
   - **Problem**: Base model's instruction-following format deeply embedded
   - **Impact**: MEDIUM - May require stronger training signal

### 1.3 Success Criteria (Corrected Stage 0)

**Format Adherence**: **>80%** (vs 0% current)
- At least 4 out of 5 test examples produce parseable 6-phase structure
- All 6 phases (Samshaya, Pramana, Pancha Avayava, Tarka, Hetvabhasa, Nirnaya) present

**Phase Completeness**: **>70%**
- Average 4.2+ phases present per output (out of 6)
- Pramana section contains at least 2 of 4 sources (Pratyaksha, Anumana, Upamana, Shabda)
- Pancha Avayava contains at least 1 complete syllogism (all 5 members)

**Answer Correctness**: **>60%**
- Model answers match ground truth in at least 3 out of 5 test examples
- Note: Format learning is primary goal; correctness secondary for Stage 0

**Training Metrics**:
- Training loss decreases smoothly (no overfitting spikes)
- Validation loss tracks training loss (if validation split added)
- Format adherence during training >50% by epoch 10

---

## 2. Corrective Actions

### 2.1 Data Augmentation Strategy

**Priority**: P0 (CRITICAL)  
**Timeline**: Week 1-2  
**Effort**: 20-30 hours manual work

#### Action 2.1.1: Expand Seed Examples from 5 to 20

**Current State**: 5 examples in `data/seed_examples/stage_zero/`
- `pramana-001-constraint.md` (Constraint Satisfaction)
- `pramana-002-boolean.md` (Boolean SAT)
- `pramana-003-transitive.md` (Transitive Reasoning)
- `pramana-004-set.md` (Set Membership)
- `pramana-005-deduction.md` (Multi-Step Deduction)

**Target State**: 20 examples (4 per problem type)

**Implementation Plan**:

1. **Create 15 new examples** following existing format:
   ```bash
   # Template for new examples
   data/seed_examples/stage_zero/pramana-006-constraint.md
   data/seed_examples/stage_zero/pramana-007-constraint.md
   data/seed_examples/stage_zero/pramana-008-constraint.md
   data/seed_examples/stage_zero/pramana-009-boolean.md
   data/seed_examples/stage_zero/pramana-010-boolean.md
   data/seed_examples/stage_zero/pramana-011-boolean.md
   data/seed_examples/stage_zero/pramana-012-transitive.md
   data/seed_examples/stage_zero/pramana-013-transitive.md
   data/seed_examples/stage_zero/pramana-014-transitive.md
   data/seed_examples/stage_zero/pramana-015-set.md
   data/seed_examples/stage_zero/pramana-016-set.md
   data/seed_examples/stage_zero/pramana-017-set.md
   data/seed_examples/stage_zero/pramana-018-deduction.md
   data/seed_examples/stage_zero/pramana-019-deduction.md
   data/seed_examples/stage_zero/pramana-020-deduction.md
   ```

2. **Quality Requirements**:
   - Each example must demonstrate complete 6-phase Nyaya structure
   - All examples verified with `pramana validate` command
   - Z3-verifiable examples marked with `z3_verifiable: true` in frontmatter
   - Format consistency: Use existing examples as templates

3. **Validation Script**:
   ```bash
   # Validate all examples before training
   python -m pramana.cli.commands.validate \
     --input data/seed_examples/stage_zero \
     --output results/stage_0_validation.json
   ```

4. **Data Preparation**:
   ```bash
   # Regenerate training data with 20 examples
   python scripts/prepare_training_data.py \
     --input data/seed_examples/stage_zero \
     --output data/training/stage_0_corrected.jsonl
   ```

**Success Criteria**:
- ✅ 20 examples created and validated
- ✅ All examples pass structural validation
- ✅ Training data JSONL contains 20 entries
- ✅ Format consistency verified across all examples

**File Paths**:
- Seed examples: `data/seed_examples/stage_zero/pramana-006.md` through `pramana-020.md`
- Training data: `data/training/stage_0_corrected.jsonl`
- Validation report: `results/stage_0_validation.json`

---

### 2.2 Training Modifications

#### Action 2.2.1: Increase LoRA Rank from 32 to 64

**Priority**: P0 (CRITICAL)  
**File**: `scripts/train_unsloth_dgx.py`  
**Line**: 84

**Current Configuration**:
```python
model = FastLanguageModel.get_peft_model(
    model,
    r=32,  # LoRA rank
    lora_alpha=32,
    ...
)
```

**Corrected Configuration**:
```python
model = FastLanguageModel.get_peft_model(
    model,
    r=64,  # LoRA rank (doubled from 32)
    lora_alpha=64,  # Match rank
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        # Consider adding for format learning:
        # "embed_tokens", "lm_head"  # Input/output embeddings
    ],
    ...
)
```

**Impact**:
- Trainable parameters: ~97M (vs 48.6M current) = ~3% of model (vs 1.49%)
- Memory increase: ~2x (still manageable on A100 40GB)
- Expected improvement: Higher capacity for format learning

**Validation**:
```python
# After model creation, verify trainable parameters
model.print_trainable_parameters()
# Expected: ~97M trainable parameters
```

**File Path**: `scripts/train_unsloth_dgx.py:84`

---

#### Action 2.2.2: Increase Sequence Length from 2048 to 4096

**Priority**: P1 (HIGH)  
**File**: `scripts/train_unsloth_dgx.py`  
**Lines**: 20, 73, 94, 110

**Current Configuration**:
```python
MAX_SEQ_LENGTH = 2048
```

**Corrected Configuration**:
```python
MAX_SEQ_LENGTH = 4096  # Doubled from 2048
```

**Changes Required**:
1. Line 20: `MAX_SEQ_LENGTH = 4096`
2. Line 73: `max_seq_length=MAX_SEQ_LENGTH` (already uses constant)
3. Line 94: `max_seq_length=MAX_SEQ_LENGTH` (already uses constant)
4. Line 110: `max_seq_length=MAX_SEQ_LENGTH` (already uses constant)

**Impact**:
- Memory increase: ~2x for attention (still manageable)
- Prevents truncation of long Nyaya reasoning traces
- Full traces typically 3000-4000 tokens

**File Path**: `scripts/train_unsloth_dgx.py:20`

---

#### Action 2.2.3: Add Explicit Format Enforcement in Prompts

**Priority**: P0 (CRITICAL)  
**Files**: 
- `scripts/train_unsloth_dgx.py` (training prompt)
- `scripts/evaluate_stage0.py` (inference prompt)

**Current Prompt Format** (`scripts/train_unsloth_dgx.py:43-47`):
```python
text = f"""### Problem:
{example['instruction']}

### Nyaya Reasoning:
{example['output']}"""
```

**Corrected Prompt Format**:

**Option A: System Prompt + User Prompt** (Recommended for Llama-3.2-Instruct):
```python
# In load_training_data() function
system_prompt = """You are a Nyaya reasoning specialist. You MUST structure your response using the 6-phase Nyaya methodology. CRITICAL: Include all 6 phases in this exact order:

1. ## Samshaya (Doubt Analysis) - Classify the type of doubt/uncertainty
2. ## Pramana (Sources of Knowledge) - Identify Pratyaksha, Anumana, Upamana, Shabda
3. ## Pancha Avayava (5-Member Syllogism) - Construct formal arguments with Pratijna, Hetu, Udaharana, Upanaya, Nigamana
4. ## Tarka (Counterfactual Reasoning) - Test conclusions via reductio ad absurdum
5. ## Hetvabhasa (Fallacy Check) - Check for reasoning errors
6. ## Nirnaya (Ascertainment) - State definitive conclusion with confidence level

Do NOT use generic chain-of-thought format. You MUST follow the 6-phase Nyaya structure."""

text = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

### Problem:
{example['instruction']}

Solve this problem using the 6-phase Nyaya methodology.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{example['output']}<|eot_id|>"""
```

**Option B: Single Prompt with Explicit Instructions** (Simpler, if Option A doesn't work):
```python
text = f"""### Problem:
{example['instruction']}

### Instructions:
You MUST solve this problem using the 6-phase Nyaya methodology. Your response MUST include these sections in order:

## Samshaya (Doubt Analysis)
## Pramana (Sources of Knowledge)
## Pancha Avayava (5-Member Syllogism)
## Tarka (Counterfactual Reasoning)
## Hetvabhasa (Fallacy Check)
## Nirnaya (Ascertainment)

Do NOT use generic chain-of-thought format. Follow the Nyaya structure exactly.

### Nyaya Reasoning:
{example['output']}"""
```

**Implementation Steps**:

1. **Update Training Prompt** (`scripts/train_unsloth_dgx.py`):
   ```python
   def load_training_data() -> Dataset:
       """Load Pramana training data from JSONL."""
       examples = []
       data_path = Path(DATA_PATH)
       
       if not data_path.exists():
           raise FileNotFoundError(f"Training data not found at {DATA_PATH}")
       
       # Format enforcement prompt
       format_instructions = """You MUST structure your response using the 6-phase Nyaya methodology:

## Samshaya (Doubt Analysis)
## Pramana (Sources of Knowledge)
## Pancha Avayava (5-Member Syllogism)
## Tarka (Counterfactual Reasoning)
## Hetvabhasa (Fallacy Check)
## Nirnaya (Ascertainment)

Do NOT use generic chain-of-thought format."""
       
       with open(data_path) as f:
           for line in f:
               line = line.strip()
               if not line:
                   continue
               example = json.loads(line)
               # Format with explicit instructions
               text = f"""### Problem:
{example['instruction']}

### Instructions:
{format_instructions}

### Nyaya Reasoning:
{example['output']}"""
               examples.append({"text": text})
       
       print(f"Loaded {len(examples)} training examples")
       return Dataset.from_list(examples)
   ```

2. **Update Inference Prompt** (`scripts/evaluate_stage0.py:82-95`):
   ```python
   def create_prompt(problem: str) -> str:
       """Create inference prompt with format enforcement."""
       format_instructions = """You MUST structure your response using the 6-phase Nyaya methodology:

## Samshaya (Doubt Analysis)
## Pramana (Sources of Knowledge)
## Pancha Avayava (5-Member Syllogism)
## Tarka (Counterfactual Reasoning)
## Hetvabhasa (Fallacy Check)
## Nirnaya (Ascertainment)

Do NOT use generic chain-of-thought format."""
       
       return f"""### Problem:
{problem}

### Instructions:
{format_instructions}

### Nyaya Reasoning:
"""
   ```

**File Paths**:
- Training: `scripts/train_unsloth_dgx.py:28-51`
- Inference: `scripts/evaluate_stage0.py:82-95`

**Success Criteria**:
- ✅ Prompts include explicit format instructions
- ✅ Training and inference prompts match
- ✅ Format instructions appear before "Nyaya Reasoning" section

---

#### Action 2.2.4: Reduce Epochs and Add Validation Split

**Priority**: P1 (HIGH)  
**File**: `scripts/train_unsloth_dgx.py`

**Current Configuration**:
```python
max_steps=50,  # ~10 passes through 5 examples (25 epochs)
# No validation split
```

**Corrected Configuration**:
```python
# Split dataset: 80% train, 20% validation
from datasets import Dataset
dataset = load_training_data()
split_dataset = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,  # Add validation set
    tokenizer=tokenizer,
    args=SFTConfig(
        max_seq_length=MAX_SEQ_LENGTH,
        per_device_train_batch_size=2,  # Increased from 1
        gradient_accumulation_steps=4,
        warmup_steps=4,  # Increased from 2
        max_steps=80,  # ~10 epochs on 16 training examples
        num_train_epochs=None,  # Use max_steps instead
        eval_strategy="steps",  # Evaluate during training
        eval_steps=20,  # Evaluate every 20 steps
        save_strategy="steps",
        save_steps=20,
        logging_steps=5,
        output_dir=OUTPUT_DIR,
        optim="adamw_8bit",
        learning_rate=2e-5,
        seed=42,
        fp16=False,
        bf16=True,
        load_best_model_at_end=True,  # Load best checkpoint
        metric_for_best_model="eval_loss",  # Use validation loss
    ),
    dataset_text_field="text",
)
```

**Impact**:
- Prevents overfitting: Validation loss tracks training loss
- Early stopping: Can stop if validation loss increases
- Better generalization: Model evaluated on unseen examples during training

**File Path**: `scripts/train_unsloth_dgx.py:54-126`

**Success Criteria**:
- ✅ Validation split created (16 train, 4 validation from 20 examples)
- ✅ Evaluation runs during training
- ✅ Validation loss decreases with training loss
- ✅ Best checkpoint saved based on validation loss

---

#### Action 2.2.5: Increase Batch Size from 1 to 2

**Priority**: P2 (MEDIUM)  
**File**: `scripts/train_unsloth_dgx.py:111`

**Current Configuration**:
```python
per_device_train_batch_size=1,  # Small batch for 5 examples
```

**Corrected Configuration**:
```python
per_device_train_batch_size=2,  # Increased for 20 examples
```

**Rationale**:
- With 20 examples, batch size 2 is feasible
- Enables padding-free training benefits (Unsloth optimization)
- ~2x faster training

**File Path**: `scripts/train_unsloth_dgx.py:111`

---

### 2.3 Validation Strategy

#### Action 2.3.1: Add Format Validation During Training

**Priority**: P1 (HIGH)  
**File**: `scripts/train_unsloth_dgx.py` (new callback)

**Implementation**:

Create a training callback that validates format adherence during training:

```python
from transformers import TrainerCallback
from pramana.application.evaluation.handlers import Tier1StructuralHandler
from pramana.application.data.parser import MarkdownParser

class FormatValidationCallback(TrainerCallback):
    """Callback to validate format adherence during training."""
    
    def __init__(self, tokenizer, sample_problems: list[str], eval_steps: int = 20):
        self.tokenizer = tokenizer
        self.sample_problems = sample_problems
        self.eval_steps = eval_steps
        self.parser = MarkdownParser()
        self.tier1_handler = Tier1StructuralHandler()
        self.format_adherence_history = []
    
    def on_evaluate(self, args, state, control, model=None, **kwargs):
        """Run format validation after evaluation."""
        if state.global_step % self.eval_steps != 0:
            return
        
        # Generate outputs for sample problems
        model.eval()
        format_scores = []
        
        for problem in self.sample_problems[:2]:  # Test on 2 problems
            prompt = create_prompt(problem)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    temperature=0.7,
                    do_sample=True,
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Try to parse and validate
            try:
                # Wrap with minimal frontmatter for parser
                wrapped = f"---\nid: test\n---\n\n# Problem\n\n{problem}\n\n{generated_text}"
                parsed = self.parser.parse(wrapped)
                
                # Check format adherence
                result = self.tier1_handler.evaluate(parsed, generated_text)
                format_scores.append(1.0 if result.passed else 0.0)
            except:
                format_scores.append(0.0)
        
        avg_format_adherence = sum(format_scores) / len(format_scores) if format_scores else 0.0
        self.format_adherence_history.append({
            "step": state.global_step,
            "format_adherence": avg_format_adherence,
        })
        
        print(f"\n[Step {state.global_step}] Format Adherence: {avg_format_adherence:.2%}")
        
        # Log to wandb if available
        if hasattr(args, "report_to") and "wandb" in args.report_to:
            import wandb
            wandb.log({"format_adherence": avg_format_adherence}, step=state.global_step)
        
        model.train()
```

**Usage in Training**:
```python
# Load sample problems for validation
sample_problems = [
    "Three people (Alice, Bob, Carol) each have one pet...",
    "Given: If P then Q, If Q then R, P is true...",
]

# Add callback
format_callback = FormatValidationCallback(
    tokenizer=tokenizer,
    sample_problems=sample_problems,
    eval_steps=20,
)

trainer = SFTTrainer(
    ...,
    callbacks=[format_callback],
)
```

**File Path**: Create new file `scripts/format_validation_callback.py`

**Success Criteria**:
- ✅ Format adherence logged during training
- ✅ Format adherence >50% by epoch 10
- ✅ Format adherence increases over training steps

---

## 3. Implementation Sequence

### Phase 1: Data Preparation (Week 1, Days 1-3)

**Day 1-2: Create New Examples**
1. Create 15 new seed examples (3 per problem type)
2. Validate each example with `pramana validate`
3. Ensure format consistency with existing examples

**Day 3: Prepare Training Data**
1. Run `scripts/prepare_training_data.py` with 20 examples
2. Verify JSONL contains 20 entries
3. Validate training data format

**Deliverables**:
- ✅ 20 seed examples in `data/seed_examples/stage_zero/`
- ✅ Training data: `data/training/stage_0_corrected.jsonl`
- ✅ Validation report: `results/stage_0_validation.json`

**Commands**:
```bash
# Create new examples (manual work)
# ... create pramana-006.md through pramana-020.md ...

# Validate all examples
python -m pramana.cli.commands.validate \
  --input data/seed_examples/stage_zero \
  --output results/stage_0_validation.json

# Prepare training data
python scripts/prepare_training_data.py \
  --input data/seed_examples/stage_zero \
  --output data/training/stage_0_corrected.jsonl

# Verify training data
wc -l data/training/stage_0_corrected.jsonl  # Should be 20
```

---

### Phase 2: Code Modifications (Week 1, Days 4-5)

**Day 4: Update Training Script**
1. Increase LoRA rank: 32 → 64
2. Increase sequence length: 2048 → 4096
3. Add format enforcement prompts
4. Add validation split
5. Increase batch size: 1 → 2
6. Update training steps/epochs

**Day 5: Update Evaluation Script**
1. Add format enforcement to inference prompts
2. Ensure prompt format matches training

**Deliverables**:
- ✅ Updated `scripts/train_unsloth_dgx.py`
- ✅ Updated `scripts/evaluate_stage0.py`
- ✅ Format validation callback (optional)

**Commands**:
```bash
# Test training script (dry run)
python scripts/train_unsloth_dgx.py --dry-run  # If dry-run flag exists

# Verify changes
git diff scripts/train_unsloth_dgx.py
git diff scripts/evaluate_stage0.py
```

---

### Phase 3: Training Execution (Week 2, Days 1-2)

**Day 1: Run Training**
1. Execute corrected training script
2. Monitor format adherence during training (if callback added)
3. Verify validation loss tracks training loss

**Day 2: Evaluate Model**
1. Run evaluation script on held-out test examples
2. Calculate format adherence metrics
3. Compare to Stage 0 success criteria

**Deliverables**:
- ✅ Trained model: `models/stage_0_corrected/`
- ✅ Evaluation results: `results/stage_0_corrected_evaluation.json`
- ✅ Training logs: `results/stage_0_corrected_training.log`

**Commands**:
```bash
# Run training
python scripts/train_unsloth_dgx.py

# Evaluate model
python scripts/evaluate_stage0.py \
  --model-dir models/stage_0_corrected \
  --output results/stage_0_corrected_evaluation.json
```

---

### Phase 4: Validation and Analysis (Week 2, Days 3-5)

**Day 3-4: Analyze Results**
1. Compare format adherence: 0% → target >80%
2. Analyze phase completeness
3. Identify remaining issues

**Day 5: Iterate if Needed**
1. If format adherence <50%: Consider additional fixes (see Rollback Plan)
2. If format adherence 50-80%: Minor adjustments
3. If format adherence >80%: Stage 0 SUCCESS

**Deliverables**:
- ✅ Analysis report: `docs/stage_0_corrected_analysis.md`
- ✅ Decision: Proceed to Stage 1 or iterate

---

## 4. Validation Protocol

### 4.1 Pre-Training Validation

**Checklist**:
- [ ] 20 seed examples created and validated
- [ ] Training data JSONL contains 20 entries
- [ ] All examples pass structural validation
- [ ] LoRA rank set to 64
- [ ] Sequence length set to 4096
- [ ] Format enforcement prompts added
- [ ] Validation split configured (80/20)
- [ ] Batch size set to 2

**Validation Commands**:
```bash
# Validate examples
python -m pramana.cli.commands.validate \
  --input data/seed_examples/stage_zero \
  --output results/stage_0_validation.json

# Check training data
python -c "
import json
with open('data/training/stage_0_corrected.jsonl') as f:
    lines = [json.loads(l) for l in f]
    print(f'Total examples: {len(lines)}')
    print(f'Example keys: {list(lines[0].keys())}')
"
```

---

### 4.2 During-Training Validation

**Metrics to Monitor**:
1. **Training Loss**: Should decrease smoothly
2. **Validation Loss**: Should track training loss (no divergence)
3. **Format Adherence** (if callback added): Should increase over steps
4. **Gradient Norms**: Should remain stable (<1.0)

**Success Criteria**:
- Training loss decreases from ~1.2 to <0.8
- Validation loss tracks training loss (difference <0.2)
- Format adherence >50% by step 40 (epoch 5)
- No gradient explosion (all norms <1.0)

**Monitoring Commands**:
```bash
# Watch training logs
tail -f results/stage_0_corrected_training.log

# Check format adherence (if logged)
grep "Format Adherence" results/stage_0_corrected_training.log
```

---

### 4.3 Post-Training Validation

**Evaluation Protocol**:

1. **Load Model**:
   ```python
   from unsloth import FastLanguageModel
   model, tokenizer = FastLanguageModel.from_pretrained(
       "models/stage_0_corrected",
       max_seq_length=4096,
       load_in_4bit=True,
   )
   FastLanguageModel.for_inference(model)
   ```

2. **Run Evaluation**:
   ```bash
   python scripts/evaluate_stage0.py \
     --model-dir models/stage_0_corrected \
     --test-examples pramana-003 pramana-005 \
     --output results/stage_0_corrected_evaluation.json
   ```

3. **Calculate Metrics**:
   - Format adherence: % of examples with all 6 phases
   - Phase completeness: Average phases present per example
   - Answer correctness: % matching ground truth

**Success Criteria**:
- ✅ Format adherence: **>80%** (4/5 test examples)
- ✅ Phase completeness: **>70%** (average 4.2+ phases)
- ✅ Answer correctness: **>60%** (3/5 test examples)

**Validation Script**:
```python
# scripts/validate_stage0_results.py
import json

with open("results/stage_0_corrected_evaluation.json") as f:
    results = json.load(f)

format_adherence = sum(
    1 for r in results["results"]
    if r.get("format_metrics", {}).get("num_phases_present", 0) == 6
) / len(results["results"])

print(f"Format Adherence: {format_adherence:.1%}")
print(f"Target: >80%")
print(f"Status: {'✓ PASS' if format_adherence >= 0.8 else '✗ FAIL'}")
```

---

## 5. Rollback Plan

### 5.1 If Format Adherence <30% After Corrections

**Symptom**: Model still produces generic CoT after all corrections.

**Possible Causes**:
1. Base model too strong (Llama-3.2-Instruct CoT priors)
2. Format enforcement insufficient
3. Training signal too weak

**Rollback Actions**:

**Option A: Switch Base Model** (Priority 1)
- **Action**: Use `unsloth/DeepSeek-R1-Distill-Llama-8B-bnb-4bit` instead
- **Rationale**: Pre-trained with reasoning traces, may adapt better
- **Implementation**:
  ```python
  model, tokenizer = FastModel.from_pretrained(
      model_name="unsloth/DeepSeek-R1-Distill-Llama-8B-bnb-4bit",
      ...
  )
  ```
- **File**: `scripts/train_unsloth_dgx.py:71`

**Option B: Increase LoRA Rank to 128** (Priority 2)
- **Action**: Double LoRA rank again (64 → 128)
- **Rationale**: Even higher capacity for format learning
- **Implementation**:
  ```python
  r=128,  # Increased from 64
  lora_alpha=128,
  ```
- **File**: `scripts/train_unsloth_dgx.py:84`
- **Impact**: ~4x memory (still manageable on A100)

**Option C: Multi-Stage Training** (Priority 3)
- **Action**: Train in stages (Samshaya → Samshaya+Pramana → Full 6-phase)
- **Rationale**: Gradual format learning may be more effective
- **Implementation**: See `docs/implementation_report.md:1149-1152`
- **Effort**: HIGH (requires custom training loop)

**Option D: Constrained Decoding** (Priority 4)
- **Action**: Use LogitsProcessor to enforce phase transitions
- **Rationale**: Force model to follow structure during generation
- **Implementation**: See `docs/implementation_report.md:1140-1147`
- **Effort**: MEDIUM (requires custom generation logic)

**Decision Tree**:
```
Format Adherence <30%?
├─ Try Option A (Switch Base Model) → If still <30%, try Option B
├─ Try Option B (Increase Rank to 128) → If still <30%, try Option C
└─ Try Option C (Multi-Stage Training) → If still <30%, reconsider approach
```

---

### 5.2 If Format Adherence 30-50% After Corrections

**Symptom**: Partial format learning (some phases present, but incomplete).

**Possible Causes**:
1. Format enforcement needs strengthening
2. More training data needed
3. Training not converged

**Rollback Actions**:

**Option A: Strengthen Format Prompts** (Priority 1)
- **Action**: Add more explicit phase-by-phase instructions
- **Implementation**: Expand format instructions with examples:
  ```python
  format_instructions = """You MUST structure your response using the 6-phase Nyaya methodology. Each phase MUST appear with its exact header:

## Samshaya (Doubt Analysis)
[Your doubt analysis here]

## Pramana (Sources of Knowledge)
### Pratyaksha (Direct Perception)
[Observable facts]
### Anumana (Inference)
[Inferences]
### Upamana (Comparison)
[Comparisons]
### Shabda (Testimony)
[Authoritative sources]

## Pancha Avayava (5-Member Syllogism)
### Syllogism 1: [Title]
**Pratijna (Thesis)**: [Thesis statement]
**Hetu (Reason)**: [Reason]
**Udaharana (Universal + Example)**: [Universal rule with example]
**Upanaya (Application)**: [Application to specific case]
**Nigamana (Conclusion)**: [Conclusion]

## Tarka (Counterfactual Reasoning)
[Reductio ad absurdum testing]

## Hetvabhasa (Fallacy Check)
[Fallacy checking]

## Nirnaya (Ascertainment)
[Definitive conclusion with confidence]

CRITICAL: Include ALL 6 phases. Do NOT skip any phase."""
  ```

**Option B: Increase Training Data to 30 Examples** (Priority 2)
- **Action**: Create 10 more examples (total 30)
- **Rationale**: More format repetition
- **Effort**: 10-15 hours manual work

**Option C: Train Longer** (Priority 3)
- **Action**: Increase max_steps to 120 (15 epochs)
- **Rationale**: Model may need more training
- **Implementation**: `max_steps=120` in training config

---

### 5.3 If Format Adherence 50-80% After Corrections

**Symptom**: Good format learning, but not meeting >80% target.

**Possible Causes**:
1. Edge cases not covered in training
2. Some problem types harder than others
3. Minor prompt adjustments needed

**Rollback Actions**:

**Option A: Fine-Tune Prompts** (Priority 1)
- **Action**: Adjust format instructions based on failure patterns
- **Rationale**: Targeted improvements for specific phases

**Option B: Add More Examples for Problem Types** (Priority 2)
- **Action**: Identify which problem types fail, add more examples
- **Rationale**: Better coverage for difficult types

**Option C: Accept 50-80% as Stage 0 Success** (Priority 3)
- **Action**: Proceed to Stage 1 with current model
- **Rationale**: Format learning demonstrated, can improve in Stage 1
- **Decision**: Requires project lead approval

---

### 5.4 If Training Fails (Memory/Compute Issues)

**Symptom**: Out-of-memory errors or training crashes.

**Rollback Actions**:

**Option A: Reduce Sequence Length** (Priority 1)
- **Action**: 4096 → 3072 tokens
- **Rationale**: Lower memory usage
- **Trade-off**: May truncate some long traces

**Option B: Reduce Batch Size** (Priority 2)
- **Action**: 2 → 1
- **Rationale**: Lower memory per step
- **Trade-off**: Slower training, less efficient

**Option C: Reduce LoRA Rank** (Priority 3)
- **Action**: 64 → 48 (compromise)
- **Rationale**: Lower memory usage
- **Trade-off**: Lower capacity

**Option D: Use Gradient Checkpointing** (Priority 4)
- **Action**: Already enabled (`use_gradient_checkpointing="unsloth"`)
- **Rationale**: Reduces memory by 30%
- **Status**: Already implemented

---

## 6. Success Metrics and Decision Criteria

### 6.1 Stage 0 Success Criteria (Corrected)

**Primary Metric: Format Adherence**
- **Target**: >80% (4/5 test examples produce parseable 6-phase structure)
- **Measurement**: Count examples with all 6 phases present and parseable
- **Current**: 0% (baseline)

**Secondary Metrics**:

1. **Phase Completeness**
   - **Target**: Average 4.2+ phases present per example (out of 6)
   - **Measurement**: Count phases present in each parsed output
   - **Current**: 0 phases (baseline)

2. **Answer Correctness**
   - **Target**: >60% (3/5 test examples match ground truth)
   - **Measurement**: Compare Nirnaya answer to ground truth
   - **Current**: 100% (but wrong format)

3. **Training Metrics**
   - **Target**: Training loss <0.8, validation loss tracks training loss
   - **Measurement**: Monitor loss curves during training
   - **Current**: Final loss 0.9898 (baseline)

### 6.2 Decision Criteria

**Proceed to Stage 1 If**:
- ✅ Format adherence: **>80%**
- ✅ Phase completeness: **>70%** (average 4.2+ phases)
- ✅ Answer correctness: **>60%**
- ✅ Training loss: **<0.8**
- ✅ Validation loss: **Tracks training loss** (difference <0.2)

**Iterate Stage 0 If**:
- ❌ Format adherence: **<50%**
- ❌ Phase completeness: **<50%** (average <3 phases)
- ❌ Training loss: **>1.0** or diverging
- ❌ Validation loss: **Diverging from training loss** (>0.5 difference)

**Consider Alternative Approach If**:
- ❌ Format adherence: **<30%** after all corrections
- ❌ Model produces identical output to base model
- ❌ No improvement after 2 iterations

---

## 7. Implementation Checklist

### Pre-Implementation
- [ ] Review this corrective plan
- [ ] Confirm resource availability (DGX Spark access, time for manual example creation)
- [ ] Backup current `models/stage_0/` directory
- [ ] Create git branch: `fix/stage-0-format-learning`

### Phase 1: Data Preparation
- [ ] Create 15 new seed examples (pramana-006 through pramana-020)
- [ ] Validate all 20 examples with `pramana validate`
- [ ] Regenerate training data JSONL
- [ ] Verify training data contains 20 entries

### Phase 2: Code Modifications
- [ ] Update `scripts/train_unsloth_dgx.py`:
  - [ ] LoRA rank: 32 → 64
  - [ ] Sequence length: 2048 → 4096
  - [ ] Add format enforcement prompts
  - [ ] Add validation split (80/20)
  - [ ] Batch size: 1 → 2
  - [ ] Update training steps/epochs
- [ ] Update `scripts/evaluate_stage0.py`:
  - [ ] Add format enforcement to inference prompts
- [ ] (Optional) Create format validation callback

### Phase 3: Training Execution
- [ ] Run corrected training script
- [ ] Monitor training metrics (loss, format adherence if callback added)
- [ ] Verify validation loss tracks training loss
- [ ] Save trained model to `models/stage_0_corrected/`

### Phase 4: Evaluation
- [ ] Run evaluation script on test examples
- [ ] Calculate format adherence metrics
- [ ] Compare to success criteria
- [ ] Generate evaluation report

### Post-Implementation
- [ ] Document results in `docs/stage_0_corrected_results.md`
- [ ] Compare to baseline (0% format adherence)
- [ ] Make go/no-go decision for Stage 1
- [ ] If successful: Merge branch and update `CLAUDE.md` with corrected Stage 0 status
- [ ] If failed: Execute rollback plan and iterate

---

## 8. File Reference

### Modified Files

1. **Training Script**: `scripts/train_unsloth_dgx.py`
   - Lines 20, 43-47, 73, 84, 94, 100-126

2. **Evaluation Script**: `scripts/evaluate_stage0.py`
   - Lines 82-95

3. **Data Preparation**: `scripts/prepare_training_data.py`
   - No changes required (already handles multiple examples)

### New Files

1. **Format Validation Callback** (optional): `scripts/format_validation_callback.py`

2. **Corrected Training Data**: `data/training/stage_0_corrected.jsonl`

3. **New Seed Examples**: `data/seed_examples/stage_zero/pramana-006.md` through `pramana-020.md`

### Output Files

1. **Trained Model**: `models/stage_0_corrected/`

2. **Evaluation Results**: `results/stage_0_corrected_evaluation.json`

3. **Training Logs**: `results/stage_0_corrected_training.log`

4. **Analysis Report**: `docs/stage_0_corrected_analysis.md`

---

## 9. Timeline and Effort Estimate

### Total Timeline: 2 Weeks

**Week 1**:
- Days 1-3: Data preparation (20-30 hours manual work)
- Days 4-5: Code modifications (4-6 hours)

**Week 2**:
- Days 1-2: Training execution (2-4 hours compute time)
- Days 3-5: Evaluation and analysis (4-6 hours)

**Total Effort**: ~30-46 hours

### Resource Requirements

- **Compute**: DGX Spark (1 GPU, ~2-4 hours training time)
- **Storage**: ~2GB for model checkpoints
- **Manual Work**: 20-30 hours for creating 15 new seed examples

---

## 10. Risk Assessment

### High-Risk Items

1. **Manual Example Creation** (20-30 hours)
   - **Risk**: Time-consuming, may delay timeline
   - **Mitigation**: Prioritize quality over quantity, use templates

2. **Format Learning Still Fails** (<30% adherence)
   - **Risk**: May require base model switch or alternative approach
   - **Mitigation**: Rollback plan includes multiple options

3. **Memory/Compute Issues**
   - **Risk**: Increased LoRA rank and sequence length may cause OOM
   - **Mitigation**: Rollback plan includes memory reduction options

### Medium-Risk Items

1. **Prompt Format Mismatch**
   - **Risk**: Training and inference prompts don't match
   - **Mitigation**: Use same format function for both

2. **Overfitting on 20 Examples**
   - **Risk**: Still small dataset, may overfit
   - **Mitigation**: Validation split, early stopping, monitor validation loss

---

## 11. Conclusion

This corrective action plan addresses the root causes of Stage 0 format learning failure:

1. **Data**: Expand from 5 to 20 examples (4x increase)
2. **Capacity**: Increase LoRA rank from 32 to 64 (2x increase)
3. **Format Enforcement**: Add explicit format instructions in prompts
4. **Training**: Add validation split, reduce epochs, increase batch size
5. **Sequence Length**: Increase from 2048 to 4096 tokens

**Expected Outcome**: Format adherence increases from **0%** to **>80%**, achieving Stage 0 success criteria.

**Next Steps**: Execute implementation sequence, validate results, and proceed to Stage 1 if successful.

---

**Document Version**: 1.0  
**Last Updated**: 2026-01-31  
**Status**: Ready for Implementation
