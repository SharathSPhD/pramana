# Pramana Technical Inventory

**Generated:** 2026-02-03  
**Project:** Pramana - Epistemic Reasoning Engine based on Navya-Nyaya Logic  
**Version:** 0.1.0

## Table of Contents

1. [Source Code Structure (`src/pramana/`)](#1-source-code-structure-srcpramana)
2. [Training Scripts (`scripts/`)](#2-training-scripts-scripts)
3. [Data Structures and Formats (`data/`)](#3-data-structures-and-formats-data)
4. [Model Artifacts and Results](#4-model-artifacts-and-results)
5. [Infrastructure Setup](#5-infrastructure-setup)
6. [Configuration Files](#6-configuration-files)
7. [Dependencies and Tech Stack](#7-dependencies-and-tech-stack)

---

## 1. Source Code Structure (`src/pramana/`)

### 1.1 Application Layer (`application/`)

#### 1.1.1 Data Processing (`application/data/`)

**File:** `src/pramana/application/data/parser.py`

**Purpose:** Parses structured markdown files with YAML frontmatter into NyayaExample domain models.

**Key Classes:**
- `MarkdownParser`: Main parser class
  - `parse(markdown_content: str) -> NyayaExample`: Parses markdown to domain model
  - `_extract_frontmatter()`: Extracts YAML metadata
  - `_extract_problem()`: Extracts problem statement
  - `_extract_samshaya()`, `_extract_pramana()`, etc.: Extract each Nyaya phase

**Dependencies:** `pramana.domain.models.nyaya_example`, `yaml`, `pydantic`

---

#### 1.1.2 Training (`application/training/`)

**File:** `src/pramana/application/training/base.py`

**Purpose:** Template Method pattern base class for training workflows.

**Key Classes:**
- `BaseTrainer` (ABC): Abstract base trainer
  - `train(config: StageConfig) -> TrainingResult`: Template method orchestrating training
  - `_setup()`: Abstract - load model, configure tokenizer
  - `_prepare_data()`: Abstract - load and preprocess data
  - `_train()`: Abstract - execute training loop
  - `_cleanup()`: Optional cleanup hook

- `TrainingResult` (dataclass):
  - `final_loss: float`
  - `best_checkpoint_path: Path`
  - `metrics: dict[str, float]`
  - `training_time_seconds: float`

**File:** `src/pramana/application/training/sft.py`

**Purpose:** Supervised Fine-Tuning trainer for Stages 0-2.

**Key Classes:**
- `SupervisedFineTuningTrainer(BaseTrainer)`:
  - Uses `UnslothAdapter` for model operations
  - Uses `CheckpointRepository` for saving models
  - Formats data with Nyaya prompt template
  - Integrates with `SFTTrainer` from TRL
  - Supports W&B logging

**Key Methods:**
- `_setup(config)`: Loads model via Unsloth, applies LoRA
- `_prepare_data()`: Loads JSONL or directory datasets, formats with `_format_nyaya_example()`
- `_train()`: Creates `SFTTrainer`, runs training loop, saves checkpoint

**File:** `src/pramana/application/training/callbacks.py`

**Purpose:** Training callbacks for observability.

**Key Classes:**
- `NyayaMetricsCallback(TrainerCallback)`:
  - Logs format adherence, phase count, syllogism count during evaluation
  - Generates samples using validation prompt
  - Integrates with W&B for metrics logging

**Key Methods:**
- `on_evaluate()`: Called during evaluation, generates sample and computes metrics
- `_compute_metrics()`: Calculates format adherence (0-1), phase count (0-6), syllogism count

---

#### 1.1.3 Evaluation (`application/evaluation/`)

**File:** `src/pramana/application/evaluation/pipeline.py`

**Purpose:** Orchestrates multi-tier evaluation using chain-of-responsibility pattern.

**Key Classes:**
- `EvaluationPipeline`:
  - Executes handlers in sequence, stopping on first failure
  - Tracks timing and collects all tier results

**Key Methods:**
- `evaluate(example, output) -> PipelineResult`: Runs pipeline through all handlers
- `_chain_handlers()`: Links handlers together

- `PipelineResult` (dataclass):
  - `overall_passed: bool`
  - `tier_results: list[TierResult]`
  - `final_tier: int`
  - `total_duration_ms: int`

**File:** `src/pramana/application/evaluation/handlers.py`

**Purpose:** Evaluation handlers implementing chain-of-responsibility pattern.

**Key Classes:**
- `EvaluationHandler` (ABC): Abstract base handler
  - `evaluate(example, output) -> TierResult`: Abstract evaluation method
  - `_pass_to_next()`: Passes to next handler in chain

- `Tier1StructuralHandler(EvaluationHandler)`:
  - Uses `NyayaStructureValidator` to validate structure
  - Checks 6-phase completeness, Pramana validity, syllogism completeness
  - Returns score 1.0 if valid, 0.0 if invalid

**File:** `src/pramana/application/evaluation/llm_judge.py`

**Purpose:** LLM-based evaluation handler for Tier 2.

**Key Classes:**
- `Tier2LLMJudgeHandler(EvaluationHandler)`:
  - Uses LLM client (OpenAI/Anthropic) to evaluate reasoning quality
  - Scores each of 6 phases (0-10) using structured rubric
  - Calculates weighted aggregate score

**Key Components:**
- `NyayaRubric`: Rubric weights (default: 1/7 each phase)
- `LLMClient` (Protocol): Interface for LLM clients
- `JUDGE_PROMPT_TEMPLATE`: Prompt template for LLM judge

**File:** `src/pramana/application/evaluation/z3_handler.py`

**Purpose:** Z3 verification handler for Tier 3.

**Key Classes:**
- `Tier3Z3VerifierHandler(EvaluationHandler)`:
  - Extracts SMT-LIB constraints from model output
  - Uses `Z3Verifier` to verify logical validity
  - Skips if example not marked `z3_verifiable`

**Key Methods:**
- `_extract_smtlib()`: Extracts SMT-LIB from fenced code blocks or raw text

**File:** `src/pramana/application/evaluation/scoring.py`

**Purpose:** Semantic answer scoring utilities.

**Key Functions:**
- `semantic_similarity()`: Computes cosine similarity using sentence-transformers
- `token_overlap_ratio()`: Token-based overlap metric
- `score_answers()`: Comprehensive answer matching (exact, normalized, semantic)
- `wilson_interval()`: Confidence interval calculation for proportions

**File:** `src/pramana/application/evaluation/results.py`

**Purpose:** Evaluation result types.

**Key Classes:**
- `TierResult` (dataclass):
  - `tier: int`
  - `passed: bool`
  - `score: float` (0.0-1.0)
  - `details: dict[str, Any]`
  - `errors: list[str]`

**Other Files:**
- `ablation.py`: Ablation study utilities
- `content_quality.py`: Content quality evaluation
- `model_loader.py`: Model loading utilities

---

### 1.2 CLI Layer (`cli/`)

**File:** `src/pramana/cli/main.py`

**Purpose:** Main CLI application entry point using Typer.

**Commands Registered:**
- `train`: Training command
- `evaluate`: Evaluation command
- `validate`: Validation command
- `data`: Data management subcommands

**File:** `src/pramana/cli/commands/train.py`

**Purpose:** Training command implementation.

**Key Function:**
- `train(stage, config, resume)`: Loads config, initializes trainer, runs training

**File:** `src/pramana/cli/commands/evaluate.py`

**Purpose:** Evaluation command implementation.

**Key Function:**
- `evaluate(model_path, data_path, tier)`: Builds evaluation pipeline, runs evaluation

**File:** `src/pramana/cli/commands/validate.py`

**Purpose:** Validation command for Nyaya examples.

**Key Functions:**
- `validate(file, dir, strict)`: Validates markdown files using `MarkdownParser` and `NyayaStructureValidator`

**File:** `src/pramana/cli/commands/data.py`

**Purpose:** Data management commands.

**Subcommands:**
- `parse`: Parse markdown files to JSON
- `stats`: Show dataset statistics
- `split`: Create train/eval splits

---

### 1.3 Configuration Layer (`config/`)

**File:** `src/pramana/config/settings.py`

**Purpose:** Application settings using Pydantic Settings.

**Key Class:**
- `PramanaSettings(BaseSettings)`:
  - Environment variable prefix: `PRAMANA_`
  - Supports `WANDB_PROJECT`, `HF_TOKEN`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` (no prefix)
  - Fields: `data_dir`, `models_dir`, `wandb_project`, `hf_token`, `openai_api_key`, `anthropic_api_key`, `llm_provider`, `llm_max_tokens`, `llm_timeout_seconds`

**File:** `src/pramana/config/loader.py`

**Purpose:** YAML-based configuration loading with inheritance.

**Key Classes:**
- `StageConfig`: Complete stage configuration
  - `stage: int` (0-4)
  - `model: ModelConfig`
  - `lora: LoRAConfig`
  - `training: TrainingParams`
  - `data: DataConfig`
  - `evaluation: EvaluationConfig`

- `StageConfigLoader`:
  - `load(stage, config_dir) -> StageConfig`: Loads with inheritance from `base.yaml`

**Configuration Models:**
- `ModelConfig`: Model name and revision
- `LoRAConfig`: Rank, alpha, target_modules
- `TrainingParams`: Learning rate, batch size, epochs, warmup, weight decay, etc.
- `DataConfig`: Train/eval paths, max_length
- `EvaluationConfig`: Tier thresholds

---

### 1.4 Domain Layer (`domain/`)

**File:** `src/pramana/domain/validators/structure.py`

**Purpose:** Validates Nyaya example structure.

**Key Classes:**
- `NyayaStructureValidator`:
  - `validate(example) -> ValidationResult`: Validates structure
  - `_validate_phase_completeness()`: Checks all 6 phases present
  - `_validate_pramana()`: Checks at least one knowledge source
  - `_validate_syllogisms()`: Checks all 5 members in each syllogism

- `ValidationResult` (dataclass):
  - `is_valid: bool`
  - `errors: list[ValidationError]`
  - `warnings: list[ValidationWarning]`

**File:** `src/pramana/domain/rewards/components.py`

**Purpose:** Individual reward components for GRPO training.

**Key Classes:**
- `RewardComponent` (ABC): Base class
- `FormatRewardComponent`: Checks 6-phase structure (0-1 score)
- `ValidityRewardComponent`: Logical validity heuristics (placeholder for Z3)
- `ConsistencyRewardComponent`: Internal consistency checks
- `CorrectnessRewardComponent`: Ground truth matching
- `StyleRewardComponent`: Verbosity and clarity checks

**File:** `src/pramana/domain/rewards/composite.py`

**Purpose:** Composite reward function combining multiple components.

**Key Classes:**
- `CompositeRewardFunction`:
  - Combines format, validity, consistency, correctness, style rewards
  - Normalizes to [-1, 1] range for GRPO
  - Uses configurable `RewardWeights`

- `RewardWeights` (dataclass): Default weights sum to 1.0
- `RewardResult` (dataclass): Total reward and component scores

**Domain Models:**
- Imported from `pramana.domain.models.nyaya_example`:
  - `NyayaExample`: Main domain model with 6 phases
  - `Samshaya`, `Pramana`, `PanchaAvayava`, `Tarka`, `Hetvabhasa`, `Nirnaya`
  - `DoubtType`, `HetvabhasaType` (enums)
  - `ExampleMetadata`

---

### 1.5 Infrastructure Layer (`infrastructure/`)

**File:** `src/pramana/infrastructure/ml/unsloth_adapter.py`

**Purpose:** Adapter for Unsloth's FastLanguageModel API.

**Key Classes:**
- `UnslothAdapter`:
  - `load_model(model_name, quantization) -> (model, tokenizer)`: Loads via Unsloth
  - `apply_lora(model, config) -> model`: Applies LoRA adapters

**File:** `src/pramana/infrastructure/verification/z3_verifier.py`

**Purpose:** Z3 solver adapter for SMT-LIB constraint verification.

**Key Classes:**
- `Z3Verifier`:
  - `verify(constraints, expected) -> VerificationResult`: Verifies SMT-LIB constraints
  - `_extract_model()`: Extracts model values from Z3 model

- `VerificationResult` (dataclass):
  - `is_valid: bool`
  - `is_satisfiable: bool`
  - `model: dict[str, Any] | None`
  - `execution_time_ms: int`
  - `error: str | None`

**File:** `src/pramana/infrastructure/llm/client.py`

**Purpose:** LLM client implementations for Tier 2 judge.

**Key Classes:**
- `OpenAILLMClient`: OpenAI API client
- `AnthropicLLMClient`: Anthropic API client
- `create_llm_client(settings) -> LLMClient`: Factory function

**File:** `src/pramana/infrastructure/storage/checkpoint_repository.py`

**Purpose:** Checkpoint repository for saving/loading models.

**Key Classes:**
- `CheckpointRepository`:
  - `save(checkpoint_id, model, tokenizer, metadata) -> Path`: Saves checkpoint
  - Base directory management

- `CheckpointMetadata` (Pydantic): Checkpoint metadata model

**File:** `src/pramana/infrastructure/storage/hf_uploader.py`

**Purpose:** HuggingFace Hub uploader for models and datasets.

**Key Classes:**
- `HuggingFaceUploader`:
  - `upload_model(model_path, repo_id, private) -> str`: Uploads model to HF Hub
  - `upload_dataset(data_path, repo_id, private) -> str`: Uploads dataset to HF Hub
  - `_ensure_readme()`: Generates README.md if missing

---

## 2. Training Scripts (`scripts/`)

### 2.1 Stage 0 Training

**File:** `scripts/train_stage0.py`

**Purpose:** Stage 0 proof-of-concept training script.

**Key Features:**
- Uses `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (non-gated, small model)
- Standard HuggingFace transformers (not Unsloth) for compatibility
- LoRA rank: 32, alpha: 32
- Target modules: `["q_proj", "k_proj", "v_proj", "o_proj"]`
- CPU-compatible training (float32, no bf16/fp16)
- Batch size: 1, gradient accumulation: 2
- Epochs: 2 (reduced for CPU proof-of-concept)
- Max sequence length: 2048

**Data Format:**
- Loads from `data/training/stage_0.jsonl`
- Formats as: `### Problem:\n{instruction}\n\n### Nyaya Reasoning:\n{output}`

**Output:** `models/stage_0/`

---

**File:** `scripts/train_stage0_corrected.py`

**Purpose:** Corrected version of Stage 0 training (likely fixes from initial version).

---

### 2.2 Stage 1 Training

**File:** `scripts/train_stage1.py`

**Purpose:** Stage 1 training script with Unsloth on DGX Spark.

**Key Features:**
- Uses `unsloth/DeepSeek-R1-Distill-Llama-8B-bnb-4bit`
- Unsloth optimizations (FastLanguageModel, FastModel)
- LoRA rank: 64 (configurable via `LORA_RANK` env var)
- Target modules: All linear layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`)
- Max sequence length: 4096 (configurable)
- Epochs: 10 (configurable)
- Batch size: 1, gradient accumulation: 4 (configurable)
- bf16 enabled, optim: `adamw_8bit`
- Uses `NyayaMetricsCallback` for format adherence tracking

**Data Sources:**
- `data/seed_examples/stage_zero/`
- `data/seed_examples/stage_one/`
- 80/20 train/validation split

**Prompt Format:**
- System prompt: "You are a Nyaya reasoning engine..."
- User prompt includes problem, format instructions, and template
- Uses chat template from tokenizer

**Output:** `models/stage_1/` (configurable via `OUTPUT_DIR`)

---

**File:** `scripts/train_unsloth_dgx.py`

**Purpose:** Generic Unsloth training script for DGX infrastructure.

---

### 2.3 Data Preparation

**File:** `scripts/prepare_training_data.py`

**Purpose:** Converts markdown seed examples to JSONL format.

**Key Features:**
- Parses markdown files with YAML frontmatter
- Extracts problem statement and full reasoning trace
- Outputs JSONL with `instruction`, `input`, `output` fields

**Usage:**
```bash
python scripts/prepare_training_data.py --input data/seed_examples/stage_zero --output data/training/stage_0.jsonl
```

---

**File:** `scripts/prepare_stage1_training_data.py`

**Purpose:** Stage 1-specific data preparation (likely combines stage 0 and stage 1 examples).

---

### 2.4 Evaluation Scripts

**File:** `scripts/evaluate_stage0.py`

**Purpose:** Evaluates Stage 0 model on test examples.

**Key Features:**
- Loads model from `models/stage_0`
- Tests on validation examples
- Generates outputs and evaluates format adherence
- Saves results to JSON

---

**File:** `scripts/shortcut_detection.py`

**Purpose:** Detects shortcut learning (memorization vs. reasoning).

---

**File:** `scripts/run_tier2_judge.py`

**Purpose:** Runs Tier 2 LLM judge evaluation.

---

## 3. Data Structures and Formats (`data/`)

### 3.1 Seed Examples

**Location:** `data/seed_examples/`

**Structure:**
- `stage_zero/`: 20 markdown files (pramana-001 through pramana-020)
- `stage_one/`: 30 markdown files (stage1-001 through stage1-030) + 5 negative examples

**File Format:** Markdown with YAML frontmatter

**Example Structure:**
```yaml
---
id: pramana-001
problem_type: constraint_satisfaction
difficulty: medium
variables: 3
ground_truth: "Alice has the fish, Bob has the dog, Carol has the cat"
metadata:
  stage: 0
  verified: true
  created_at: 2026-01-31
  z3_verifiable: true
---

# Problem
[Problem statement]

## Samshaya (Doubt Analysis)
[Doubt type and justification]

## Pramana (Sources of Knowledge)
### Pratyaksha (Direct Perception)
[Observable facts]

### Anumana (Inference)
[Inferences]

### Upamana (Comparison)
[Analogies]

### Shabda (Testimony)
[Principles]

## Pancha Avayava (5-Member Syllogism)
### Syllogism 1:
**Pratijna (Thesis)**: ...
**Hetu (Reason)**: ...
**Udaharana (Universal + Example)**: ...
**Upanaya (Application)**: ...
**Nigamana (Conclusion)**: ...

## Tarka (Counterfactual Reasoning)
**Hypothesis**: ...
**Consequence**: ...
**Analysis**: ...
**Resolution**: ...

## Hetvabhasa (Fallacy Check)
[Fallacy checks]

## Nirnaya (Ascertainment)
**Final Answer**: ...
**Justification**: ...
**Confidence**: ...
```

**Problem Types:**
- `constraint_satisfaction`: Assignment/constraint problems
- `boolean`: Boolean SAT problems
- `transitive`: Transitive relation problems
- `set`: Set theory problems
- `deduction`: Multi-step deduction problems

---

### 3.2 Training Data

**Location:** `data/training/`

**Files:**
- `stage_0.jsonl`: JSONL format with `instruction`, `input`, `output` fields
- `stage_1.jsonl`: Combined stage 0 and stage 1 examples

**Format:**
```json
{
  "instruction": "Problem statement",
  "input": "",
  "output": "Full Nyaya reasoning trace from ## Samshaya to ## Nirnaya"
}
```

---

### 3.3 Validation Data

**Location:** `data/validation/`

**Structure:**
- `stage_zero/`: Test examples for Stage 0 evaluation

**Format:** Same markdown format as seed examples

---

### 3.4 Synthetic Data

**Location:** `data/synthetic/`

**Status:** Placeholder directory (`.gitkeep` only)

**Future:** Will contain GPT-4o/Claude-generated examples following seed patterns

---

## 4. Model Artifacts and Results

### 4.1 Model Checkpoints

**Location:** `models/` (not in repo, generated during training)

**Structure:**
- `stage_0/`: Stage 0 model checkpoints
- `stage_1/`: Stage 1 model checkpoints

**Contents:**
- Model weights (LoRA adapters)
- Tokenizer files
- `metadata.json`: Checkpoint metadata

---

### 4.2 HuggingFace Uploads

**Location:** `hf_upload/` and `hf_upload_full/`

**Contents:**
- Model artifacts for HF Hub upload
- Tokenizer configs, chat templates
- README files

**Models:**
- `nyaya-llama-3b-stage0-merged/`: Stage 0 merged model
- `nyaya-deepseek-8b-stage1-merged/`: Stage 1 merged model

---

### 4.3 Evaluation Results

**Location:** `results/`

**Files:**
- `stage_0_evaluation.json`: Stage 0 evaluation results
- `stage_0_corrected_evaluation*.json`: Multiple corrected evaluation runs
- `stage_0_final_validation.json`: Final validation results
- `stage_0_shortcut_detection.json`: Shortcut detection analysis
- `stage_1_evaluation.json`: Stage 1 evaluation results

**Format:**
```json
{
  "evaluation_timestamp": "2026-01-31T19:04:06.435707",
  "model_dir": "models/stage_0",
  "test_examples": ["pramana-003", "pramana-005"],
  "results": [
    {
      "example_id": "pramana-003",
      "timestamp": "...",
      "problem": "...",
      "ground_truth": "...",
      "prompt": "...",
      "generated_output": "...",
      "output_length": 2964,
      "parse_success": false,
      "parse_error": "Missing required section: Samshaya",
      "evaluation": {
        "overall_passed": false,
        "final_tier": 0,
        "total_duration_ms": 0,
        "tier_results": []
      },
      "success": true
    }
  ]
}
```

---

## 5. Infrastructure Setup

### 5.1 Docker Configuration

**File:** `Dockerfile`

**Base Image:** `nvcr.io/nvidia/pytorch:24.09-py3`

**Key Features:**
- Installs `uv` package manager
- Sets up Python environment with dependencies
- Working directory: `/workspace/pramana`
- Caches dependency installation

**File:** `docker-compose.yml`

**Services:**
- `pramana`: Main development container
  - GPU access (NVIDIA)
  - Volume mounts: `src/`, `tests/`, `configs/`, `data/`, `models/`
  - Environment variables: `HF_TOKEN`, `WANDB_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`
  - Ports: 8888 (Jupyter), 6006 (TensorBoard)

**Volumes:**
- `huggingface_cache`: Persistent HF cache

---

### 5.2 Environment Variables

**File:** `.env.example`

**Required Variables:**
- `HF_TOKEN`: HuggingFace Hub token
- `WANDB_API_KEY`: Weights & Biases API key
- `OPENAI_API_KEY`: OpenAI API key (for Tier 2 judge)
- `ANTHROPIC_API_KEY`: Anthropic API key (alternative judge)

**Optional Variables:**
- `PRAMANA_DATA_DIR`: Data directory path
- `PRAMANA_MODELS_DIR`: Models directory path
- `PRAMANA_LOG_LEVEL`: Logging level

---

## 6. Configuration Files

### 6.1 Training Configurations

**File:** `configs/base.yaml`

**Default Configuration:**
```yaml
model:
  name: "unsloth/Llama-3.2-3B-Instruct"
  revision: "main"

lora:
  rank: 32
  alpha: 32
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

training:
  learning_rate: 2.0e-5
  batch_size: 2
  gradient_accumulation_steps: 4
  epochs: 3
  warmup_ratio: 0.1
  weight_decay: 0.01
  max_grad_norm: 1.0

data:
  max_length: 4096

evaluation:
  tier1_threshold: 0.9
  tier2_threshold: 0.7
```

**File:** `configs/stage_0.yaml`

**Stage 0 Overrides:**
```yaml
stage: 0

model:
  name: "unsloth/Llama-3.2-3B-Instruct"

training:
  epochs: 5
  batch_size: 1
  gradient_accumulation_steps: 8

data:
  train_path: "data/seed_examples/stage_zero"
  eval_path: "data/seed_examples/stage_zero"
```

---

### 6.2 Project Configuration

**File:** `pyproject.toml`

**Key Sections:**
- Project metadata (name, version, description)
- Dependencies (core, ml, verification, llm-judge, dev)
- Build system (hatchling)
- Tool configurations:
  - `pytest`: Test configuration
  - `ruff`: Linting rules
  - `mypy`: Type checking (strict mode)
  - `coverage`: Coverage settings

**Dependency Groups:**
- Core: `pydantic`, `pydantic-settings`, `pyyaml`, `click`, `rich`, `typer`
- ML: `torch`, `transformers`, `datasets`, `accelerate`, `trl`, `unsloth`, `peft`, `wandb`, `tensorboard`, `huggingface-hub`
- Verification: `z3-solver`
- LLM Judge: `openai`, `anthropic`
- Dev: `pytest`, `ruff`, `mypy`, `pre-commit`

---

## 7. Dependencies and Tech Stack

### 7.1 Core Technologies

**Python:** 3.11+ (requires `>=3.11`)

**ML Framework:**
- PyTorch 2.0+
- Transformers 4.40+
- Unsloth (for efficient fine-tuning)
- TRL 0.8+ (SFTTrainer)
- PEFT 0.18.1+ (LoRA)

**Data Processing:**
- HuggingFace Datasets 2.18+
- Sentence Transformers 5.2.2+ (for semantic similarity)

**Verification:**
- Z3 Solver 4.12+ (SMT-LIB verification)

**LLM APIs:**
- OpenAI 1.0+ (GPT-4o-mini for judge)
- Anthropic 0.18+ (Claude 3.5 Sonnet for judge)

**Experiment Tracking:**
- Weights & Biases 0.16+
- TensorBoard 2.16+

**CLI:**
- Typer 0.12.0+
- Rich 13.0+ (terminal formatting)
- Click 8.0+

**Configuration:**
- Pydantic 2.0+ (settings and models)
- PyYAML 6.0+ (config loading)

---

### 7.2 Development Tools

**Testing:**
- pytest 8.0+
- pytest-cov 4.1+
- pytest-asyncio 0.23+
- pytest-mock 3.12+

**Code Quality:**
- ruff 0.3+ (linting and formatting)
- mypy 1.9+ (type checking, strict mode)
- pre-commit 3.6+

**Package Management:**
- uv (fast Python package manager)
- hatchling (build backend)

---

### 7.3 Infrastructure

**Container Runtime:**
- Docker
- Docker Compose
- NVIDIA Container Toolkit (for GPU access)

**Base Images:**
- `nvcr.io/nvidia/pytorch:24.09-py3` (NVIDIA PyTorch container)

**Cloud Services:**
- HuggingFace Hub (model/dataset hosting)
- Weights & Biases (experiment tracking)
- OpenAI API / Anthropic API (LLM judge)

---

## 8. Key Architecture Patterns

### 8.1 Design Patterns

1. **Template Method Pattern** (`BaseTrainer`): Defines training workflow skeleton
2. **Chain of Responsibility** (`EvaluationPipeline`): Multi-tier evaluation handlers
3. **Adapter Pattern** (`UnslothAdapter`, `Z3Verifier`): Wraps external libraries
4. **Repository Pattern** (`CheckpointRepository`): Manages checkpoint persistence
5. **Factory Pattern** (`create_llm_client`): Creates LLM clients based on settings

### 8.2 Layer Architecture

```
┌─────────────────────────────────────┐
│         CLI Layer (Typer)           │
├─────────────────────────────────────┤
│      Application Layer              │
│  ┌──────────┬──────────┬──────────┐│
│  │ Training │ Evaluation│  Data   ││
│  └──────────┴──────────┴──────────┘│
├─────────────────────────────────────┤
│         Domain Layer                │
│  ┌──────────┬──────────┬──────────┐│
│  │ Validators│ Rewards │  Models  ││
│  └──────────┴──────────┴──────────┘│
├─────────────────────────────────────┤
│      Infrastructure Layer           │
│  ┌──────────┬──────────┬──────────┐│
│  │    ML    │Verification│ Storage ││
│  └──────────┴──────────┴──────────┘│
└─────────────────────────────────────┘
```

### 8.3 Data Flow

**Training Flow:**
1. Load config (`StageConfigLoader`)
2. Initialize trainer (`SupervisedFineTuningTrainer`)
3. Load model via adapter (`UnslothAdapter`)
4. Prepare data (`MarkdownParser` → formatted dataset)
5. Train (`SFTTrainer` with callbacks)
6. Save checkpoint (`CheckpointRepository`)

**Evaluation Flow:**
1. Load model and data
2. Generate outputs
3. Parse outputs (`MarkdownParser`)
4. Run evaluation pipeline (`EvaluationPipeline`)
   - Tier 1: Structural validation (`Tier1StructuralHandler`)
   - Tier 2: LLM judge (`Tier2LLMJudgeHandler`)
   - Tier 3: Z3 verification (`Tier3Z3VerifierHandler`)
5. Aggregate results

---

## 9. Essential Files for Understanding

### 9.1 Core Domain Understanding

1. **Domain Models** (`pramana.domain.models.nyaya_example`): Core NyayaExample structure
2. **Parser** (`application/data/parser.py`): How markdown → domain model
3. **Validator** (`domain/validators/structure.py`): Structure validation rules

### 9.2 Training Understanding

1. **Base Trainer** (`application/training/base.py`): Training workflow template
2. **SFT Trainer** (`application/training/sft.py`): Actual SFT implementation
3. **Unsloth Adapter** (`infrastructure/ml/unsloth_adapter.py`): Model loading/ LoRA
4. **Training Scripts** (`scripts/train_stage*.py`): Stage-specific training

### 9.3 Evaluation Understanding

1. **Pipeline** (`application/evaluation/pipeline.py`): Evaluation orchestration
2. **Handlers** (`application/evaluation/handlers.py`): Tier 1 handler
3. **LLM Judge** (`application/evaluation/llm_judge.py`): Tier 2 handler
4. **Z3 Handler** (`application/evaluation/z3_handler.py`): Tier 3 handler

### 9.4 Configuration Understanding

1. **Settings** (`config/settings.py`): Environment-based settings
2. **Config Loader** (`config/loader.py`): YAML config with inheritance
3. **Config Files** (`configs/*.yaml`): Stage configurations

---

## 10. Testing Structure

**Location:** `tests/`

**Structure:**
- `unit/`: Unit tests organized by module
  - `application/`: Training, evaluation, data tests
  - `cli/`: CLI command tests
  - `config/`: Configuration tests
  - `domain/`: Domain model/validator tests
  - `infrastructure/`: Infrastructure component tests
- `integration/`: Integration tests
- `e2e/`: End-to-end tests
- `fixtures/`: Test fixtures

**Test Markers:**
- `@pytest.mark.slow`: Slow tests
- `@pytest.mark.integration`: Integration tests
- `@pytest.mark.gpu`: GPU-required tests
- `@pytest.mark.e2e`: End-to-end tests

---

## Summary

The Pramana codebase is a well-structured research project implementing a Nyaya-based reasoning engine. Key architectural decisions:

1. **Clean Architecture**: Clear separation between CLI, Application, Domain, and Infrastructure layers
2. **Template Method Pattern**: Flexible training workflow via `BaseTrainer`
3. **Chain of Responsibility**: Multi-tier evaluation pipeline
4. **Adapter Pattern**: Clean interfaces to external libraries (Unsloth, Z3)
5. **Configuration-Driven**: YAML configs with inheritance for different stages
6. **Type Safety**: Strict mypy configuration, Pydantic models throughout
7. **Testability**: Comprehensive test structure with unit/integration/e2e tests

The project follows a staged approach (Stage 0: PoC, Stage 1: MVR, Stage 2+: Scaling) with clear separation of concerns and extensibility for future stages (RL, multi-agent protocols, etc.).
