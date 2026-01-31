# Pramana Implementation Report Outline

## 1. Executive Summary
- Project overview and goals
- Current stage (Stage 0: Proof of Concept)
- Key achievements and status
- High-level architecture snapshot

## 2. Overall Architecture and Components

### 2.1 System Architecture
- Layered architecture pattern (Domain → Application → Infrastructure → CLI)
- Component interaction diagram
- Data flow: Markdown → JSONL → Training → Evaluation

### 2.2 Core Components

#### 2.2.1 Domain Layer (`src/pramana/domain/`)
- **Validators** (`domain/validators/`)
  - `NyayaStructureValidator`: Validates 6-phase completeness, Pramana sources, syllogism structure
  - Validation rules: Phase completeness, Pramana sources, Pancha Avayava 5-member requirement
- **Rewards** (`domain/rewards/`)
  - `CompositeRewardFunction`: Weighted combination of 5 reward components
  - Reward weights: format (20%), validity (30%), consistency (20%), correctness (20%), style (10%)
  - `RewardComponent`: Base interface for individual reward calculations

#### 2.2.2 Application Layer (`src/pramana/application/`)
- **Training** (`application/training/`)
  - `BaseTrainer`: Abstract template method pattern for training orchestration
  - `SupervisedFineTuningTrainer`: SFT implementation for Stages 0-2
  - Training pipeline: Setup → Data Preparation → Training → Checkpointing
- **Evaluation** (`application/evaluation/`)
  - `EvaluationPipeline`: Chain-of-responsibility pattern for multi-tier evaluation
  - `Tier1StructuralHandler`: Automated structural validation
  - `Tier2LLMJudgeHandler`: LLM-based quality assessment (planned)
  - `Tier3ManualHandler`: Human review queue (planned)
- **Data** (`application/data/`)
  - `MarkdownParser`: Parses seed examples from Markdown with YAML frontmatter
  - Conversion pipeline: Markdown → NyayaExample domain model → JSONL training format

#### 2.2.3 Infrastructure Layer (`src/pramana/infrastructure/`)
- **ML** (`infrastructure/ml/`)
  - `UnslothAdapter`: Wrapper for Unsloth's FastLanguageModel API
  - Model loading with 4-bit quantization (QLoRA)
  - LoRA application with configurable rank/alpha/target_modules
- **Storage** (`infrastructure/storage/`)
  - `CheckpointRepository`: Manages model checkpoints with metadata
  - `CheckpointMetadata`: Git commit, stage, metrics, timestamp tracking
  - `HFUploader`: HuggingFace Hub integration (planned)
- **Verification** (`infrastructure/verification/`)
  - `Z3Verifier`: SMT-LIB constraint verification for formal logic problems
  - Timeout handling, model extraction, satisfiability checking

#### 2.2.4 Configuration (`src/pramana/config/`)
- `StageConfig`: Pydantic models for type-safe configuration
- `ConfigLoader`: Hierarchical YAML loading (base.yaml + stage-specific overrides)
- Configuration structure: model, lora, training, data, evaluation

### 2.3 Data Pipeline
- **Input Format**: Markdown files with YAML frontmatter (`data/seed_examples/stage_zero/*.md`)
- **Processing**: MarkdownParser → NyayaExample domain model → JSONL (`data/training/stage_0.jsonl`)
- **Training Format**: Instruction-response pairs with Nyaya-structured reasoning traces

## 3. Seed Examples

### 3.1 Stage 0 Seed Examples (`data/seed_examples/stage_zero/`)

#### 3.1.1 pramana-001-constraint.md
- **Type**: Constraint satisfaction
- **Purpose**: Demonstrates elimination method for assignment problems
- **Key Features**: 
  - 3 people, 3 pets assignment
  - Direct constraints + elimination reasoning
  - Z3-verifiable structure
- **Nyaya Phases**: Complete 6-phase structure with multiple syllogisms

#### 3.1.2 pramana-002-boolean.md
- **Type**: Boolean satisfiability (Knights and Knaves)
- **Purpose**: Demonstrates logical equivalence and systematic testing
- **Key Features**:
  - Knight/Knave truth-telling constraints
  - Logical equivalence mapping (X says Y ↔ (X is Knight) ↔ Y)
  - Multiple valid solutions with ground truth selection
- **Nyaya Phases**: Emphasizes Anumana (inference) and Tarka (counterfactual testing)

#### 3.1.3 pramana-003-transitive.md
- **Type**: Transitive reasoning
- **Purpose**: Demonstrates transitive property application
- **Key Features**:
  - Height ranking with pairwise comparisons
  - Transitive closure derivation
  - Complete ordering establishment
- **Nyaya Phases**: Highlights Samanyatodrishta inference type

#### 3.1.4 pramana-004-set.md
- **Type**: Set partitioning
- **Purpose**: Demonstrates constraint satisfaction with elimination
- **Key Features**:
  - Binary group assignment
  - Contradiction-based elimination
  - Graph coloring analogy (Upamana)
- **Nyaya Phases**: Emphasizes elimination method and Tarka reductio ad absurdum

#### 3.1.5 pramana-005-deduction.md
- **Type**: Logical deduction chain
- **Purpose**: Demonstrates modus ponens and transitive implication
- **Key Features**:
  - Chain of conditional statements (P → Q → R → S)
  - Step-by-step modus ponens application
  - Transitive closure verification
- **Nyaya Phases**: Highlights Purvavat inference and deductive chains

### 3.2 Seed Example Characteristics
- **Format**: All examples follow complete 6-phase Nyaya structure
- **Verification**: All marked as `z3_verifiable: true` in metadata
- **Coverage**: Diverse problem types (constraint satisfaction, Boolean SAT, transitive, set theory, deduction)
- **Quality**: Gold-standard examples with verified ground truth

## 4. Tuned vs Untuned Model Distinction

### 4.1 Base (Untuned) Models
- **Loading**: Via `UnslothAdapter.load_model()` from HuggingFace
- **Current Base**: `unsloth/Llama-3.2-3B-Instruct` (Stage 0)
- **Planned Base**: `DeepSeek-R1-Distill-Llama-8B` (Stage 1+)
- **Characteristics**:
  - No Nyaya-specific training
  - Standard instruction-following capabilities
  - No structured reasoning format enforcement

### 4.2 Fine-Tuned Models
- **Training Method**: Supervised Fine-Tuning (SFT) with LoRA adapters
- **LoRA Configuration**:
  - Rank: 32 (Stage 0), 64-128 (planned for Stage 1+)
  - Alpha: 32 (Stage 0), 16-32 (planned)
  - Target modules: All attention + FFN layers (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj)
  - Quantization: 4-bit (QLoRA)
- **Checkpoint Management**:
  - Saved via `CheckpointRepository` with metadata
  - Includes: git commit, stage, training metrics, timestamp
  - Path: `models/stage_{N}/checkpoint_{id}/`
- **Distinction Criteria**:
  - Tuned models: Generate Nyaya-structured outputs, follow 6-phase format
  - Untuned models: Standard LLM responses, no structure enforcement

### 4.3 Model Comparison Strategy
- **Format Adherence**: Measure % of outputs with complete 6-phase structure
- **Logical Validity**: Z3 verification rate for formal logic problems
- **Reasoning Quality**: LLM judge evaluation (Tier 2) for semantic correctness
- **Answer Accuracy**: Ground truth matching for final answers

## 5. Training Configuration

### 5.1 Configuration Structure
- **Base Config** (`configs/base.yaml`):
  - Model: `unsloth/Llama-3.2-3B-Instruct`
  - LoRA: rank=32, alpha=32, 7 target modules
  - Training: lr=2e-5, batch_size=2, grad_accum=4, epochs=3
  - Data: max_length=4096
  - Evaluation: tier1_threshold=0.9, tier2_threshold=0.7

### 5.2 Stage-Specific Configs
- **Stage 0** (`configs/stage_0.yaml`):
  - Extends base.yaml
  - Model: `unsloth/Llama-3.2-3B-Instruct` (smallest for POC)
  - Training: epochs=5, batch_size=1, grad_accum=8
  - Data: `data/seed_examples/stage_zero` (5 examples)

### 5.3 Training Hyperparameters
- **Learning Rate**: 2e-5 (preserves pre-trained reasoning)
- **Batch Size**: 1-2 (effective batch size 8-16 via gradient accumulation)
- **Epochs**: 3-5 (Stage 0), 10-15 (planned for Stage 1+)
- **Warmup**: 10% of training steps
- **Weight Decay**: 0.01
- **Max Grad Norm**: 1.0
- **Sequence Length**: 4096 tokens (accommodates full reasoning traces)

### 5.4 Training Infrastructure
- **Framework**: Unsloth + QLoRA (4-bit quantization)
- **Trainer**: TRL's `SFTTrainer` wrapped in `SupervisedFineTuningTrainer`
- **Experiment Tracking**: Weights & Biases integration (optional)
- **Checkpointing**: Per-epoch saves with metadata

## 6. Metrics from Latest Training Run

### 6.1 Training Metrics (To be populated from latest log)
- **Loss**: Training loss per epoch/step
- **Runtime**: Total training time
- **Throughput**: Tokens/second, examples/second
- **Memory**: Peak GPU memory usage
- **Checkpoint Info**: Best checkpoint path, final epoch

### 6.2 Evaluation Metrics (Structure in place)
- **Tier 1 (Structural)**:
  - Format adherence rate (% with complete 6 phases)
  - Pramana completeness (% with at least one knowledge source)
  - Syllogism completeness (% with all 5 members)
- **Tier 2 (LLM Judge)** - Planned:
  - Reasoning quality score
  - Nyaya adherence score
  - Component-wise scores
- **Tier 3 (Manual)** - Planned:
  - Human review queue
  - Quality annotations

### 6.3 Validation Metrics
- **Z3 Verification Rate**: % of formal logic problems passing Z3 verification
- **Ground Truth Accuracy**: % of final answers matching ground truth
- **Structure Score**: Composite score from `NyayaStructureValidator`

## 7. Verification and Validation Components

### 7.1 Structural Validation
- **Component**: `NyayaStructureValidator` (`domain/validators/structure.py`)
- **Checks**:
  1. Phase completeness: All 6 phases present (Samshaya, Pramana, Pancha Avayava, Tarka, Hetvabhasa, Nirnaya)
  2. Pramana validation: At least one knowledge source (Pratyaksha, Anumana, Upamana, or Shabda)
  3. Syllogism validation: Each Pancha Avayava has all 5 members (Pratijna, Hetu, Udaharana, Upanaya, Nigamana)
- **Output**: `ValidationResult` with errors/warnings

### 7.2 Z3 Verification
- **Component**: `Z3Verifier` (`infrastructure/verification/z3_verifier.py`)
- **Purpose**: Verify logical validity of formal logic problems
- **Process**:
  1. Parse Pratijna/Hetu/Udaharana from model output
  2. Autoformalize to SMT-LIB format (planned)
  3. Execute Z3 solver with timeout (30s default)
  4. Check satisfiability and extract model
  5. Compare with ground truth (if provided)
- **Output**: `VerificationResult` with satisfiability, model, execution time

### 7.3 Multi-Tier Evaluation Pipeline
- **Component**: `EvaluationPipeline` (`application/evaluation/pipeline.py`)
- **Architecture**: Chain-of-responsibility pattern
- **Tiers**:
  1. **Tier 1**: Automated structural validation (required, threshold: 0.9)
  2. **Tier 2**: LLM judge evaluation (if Tier 1 passes, threshold: 0.7)
  3. **Tier 3**: Manual human review (if Tier 2 passes)
- **Flow**: Execute handlers sequentially, stop on first failure
- **Output**: `PipelineResult` with tier results, overall pass/fail, timing

### 7.4 Reward Components (For GRPO Stage 3)
- **Format Reward**: Structural adherence to 6-phase format
- **Validity Reward**: Z3 verification + ground truth matching
- **Consistency Reward**: Tarka counterfactual verification via Z3
- **Correctness Reward**: Final answer accuracy
- **Style Reward**: Nyaya terminology and reasoning quality

## 8. Known Gaps and Risks

### 8.1 Implementation Gaps

#### 8.1.1 Data Pipeline
- **Gap**: Synthetic data generation not yet implemented
- **Impact**: Limited to 5 seed examples for Stage 0
- **Mitigation**: Planned for Stage 2 (200-500 synthetic examples)

#### 8.1.2 Evaluation Infrastructure
- **Gap**: Tier 2 (LLM judge) and Tier 3 (manual review) not fully implemented
- **Impact**: Only automated structural validation currently available
- **Mitigation**: Tier 2 planned for Stage 1, Tier 3 for Stage 2

#### 8.1.3 Z3 Autoformalization
- **Gap**: Manual SMT-LIB conversion, no automatic parsing from Nyaya format
- **Impact**: Z3 verification requires manual constraint extraction
- **Mitigation**: Autoformalizer component planned for Stage 2

#### 8.1.4 GRPO Training
- **Gap**: Reinforcement learning (GRPO) not yet implemented
- **Impact**: Only supervised fine-tuning available (Stages 0-2)
- **Mitigation**: GRPO planned for Stage 3 with composite reward function

#### 8.1.5 Benchmark Evaluation
- **Gap**: No integration with standard reasoning benchmarks (LogicBench, ProntoQA, RuleTaker)
- **Impact**: Limited evaluation to seed examples
- **Mitigation**: Benchmark runner planned for Stage 2

### 8.2 Technical Risks

#### 8.2.1 Syntactic Mimicry Without Semantic Reasoning
- **Risk**: Model generates correct format but logically incoherent content
- **Likelihood**: Medium (common in few-shot learning)
- **Mitigation**: 
  - Z3 verification for formal logic subset
  - Extensive human evaluation
  - Tier 2 LLM judge for semantic quality

#### 8.2.2 Domain Overfitting
- **Risk**: Works for logic puzzles but fails on broader reasoning
- **Likelihood**: Medium-High (small dataset, specific problem types)
- **Mitigation**: 
  - Early testing on diverse problem types
  - Expand to non-formal logic domains in Stage 4
  - Evaluate on general reasoning benchmarks

#### 8.2.3 Synthetic Data Poisoning
- **Risk**: Scaled generation produces subtle errors that propagate
- **Likelihood**: Medium (depends on generation quality)
- **Mitigation**: 
  - Statistical sampling + Z3 auto-verification
  - Human review of synthetic examples
  - Quality filtering pipeline

#### 8.2.4 Reasoning Overhead
- **Risk**: 6-phase structure too verbose for practical use
- **Likelihood**: Low-Medium (depends on use case)
- **Mitigation**: 
  - Track tokens/problem ratio
  - Consider abbreviated forms for simple cases
  - Optimize structure for common patterns

#### 8.2.5 Training Instability
- **Risk**: Small dataset (5 examples) may cause overfitting or instability
- **Likelihood**: High for Stage 0
- **Mitigation**: 
  - Early stopping based on validation metrics
  - Limited epochs (5 for Stage 0)
  - Expand to 50 examples for Stage 1

### 8.3 Infrastructure Risks

#### 8.3.1 GPU Dependency
- **Risk**: Training requires GPU (A100/H100), not accessible for all developers
- **Likelihood**: High (current state)
- **Mitigation**: 
  - Docker setup for DGX Spark deployment
  - CPU fallback for testing (limited functionality)
  - Cloud GPU options (HuggingFace Jobs)

#### 8.3.2 Dependency Management
- **Risk**: Complex dependencies (Unsloth, Z3, transformers, TRL) with version conflicts
- **Likelihood**: Medium
- **Mitigation**: 
  - `pyproject.toml` with pinned versions
  - Docker containerization
  - Regular dependency audits

#### 8.3.3 Checkpoint Storage
- **Risk**: Large model checkpoints consume significant storage
- **Likelihood**: Medium (4-bit quantization helps)
- **Mitigation**: 
  - HuggingFace Hub integration for remote storage
  - Checkpoint cleanup policies
  - Selective checkpoint saving

### 8.4 Research Risks

#### 8.4.1 Epistemic Framework Validity
- **Risk**: Navya-Nyaya methodology may not translate effectively to LLM training
- **Likelihood**: Low-Medium (hypothesis testing)
- **Mitigation**: 
  - Stage 0 proof-of-concept validates format learnability
  - Iterative refinement based on results
  - Comparison with baseline chain-of-thought

#### 8.4.2 Evaluation Methodology
- **Risk**: Evaluation metrics may not capture true reasoning quality
- **Likelihood**: Medium
- **Mitigation**: 
  - Multi-tier evaluation (automated + LLM + human)
  - Benchmark comparison with standard datasets
  - Qualitative analysis of reasoning traces

## 9. Next Steps and Roadmap

### 9.1 Immediate (Stage 0 Completion)
- Complete training run with 5 seed examples
- Evaluate format adherence on held-out examples
- Document training metrics and results

### 9.2 Short-term (Stage 1: MVP)
- Expand to 50 gold-standard examples
- Fine-tune DeepSeek-R1-Distill-Llama-8B
- Implement Tier 2 LLM judge evaluation
- Target: >90% format adherence, 60-70% accuracy

### 9.3 Medium-term (Stage 2: Scaling)
- Generate 200-500 synthetic examples
- Implement Z3 autoformalization
- Benchmark evaluation on LogicBench, ProntoQA
- Runtime Z3 verification pipeline

### 9.4 Long-term (Stage 3: RL Enhancement)
- Implement GRPO training with composite rewards
- Train Process Reward Model or use GPT-4 as judge
- Success criteria: High accuracy + systematic reasoning traces

## 10. Appendices

### 10.1 File Structure Reference
- Key file paths and their purposes
- Component location guide

### 10.2 Configuration Examples
- Complete config file examples
- Environment variable setup

### 10.3 Training Scripts
- `scripts/train_stage0.py`: Stage 0 training script
- `scripts/train_unsloth_dgx.py`: DGX Spark training script
- `scripts/prepare_training_data.py`: Data preparation pipeline

### 10.4 Glossary
- Nyaya terminology definitions
- Technical terms and abbreviations
