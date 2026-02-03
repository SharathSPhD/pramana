# Stage 1 Comprehensive Implementation Report

**Date**: 2026-02-02  
**Scope**: Stage 1 minimum viable reasoner implementation, evaluation, and deployment (HF models + Space + Ollama)  
**Status**: Training and deployment complete; evaluation shows partial format adherence and strong semantic correctness

---

## 1) Executive Summary

Stage 1 expanded the dataset, increased model capacity (DeepSeek 8B), and added observability during training. The pipeline produced a Stage 1 LoRA adapter and a full merged model, which were published to Hugging Face and deployed in the demo Space alongside Stage 0. Evaluation across 10 held-out examples shows:

- **Format adherence**: 0.40 (4/10), with 95% CI [0.168, 0.687]
- **Semantic answer correctness**: 1.00 (10/10), with 95% CI [0.510, 1.0]
- **Primary failure mode**: format parsing errors (missing sections or invalid doubt types), not answer accuracy

Deployment artifacts include:
- HF adapter: `qbz506/nyaya-deepseek-8b-stage1`
- HF full merged model: `qbz506/nyaya-deepseek-8b-stage1-full`
- HF dataset: `qbz506/pramana-nyaya-stage1`
- Demo Space with Stage 0/Stage 1 selector: `qbz506/pramana-nyaya-demo`

---

## 2) Stage 1 Objectives

**Objective**: Build a minimum viable reasoner that preserves Nyaya structure and improves reasoning quality beyond Stage 0.

**Success criteria (from project plan)**:
- >= 90% format adherence
- 60% to 70% answer correctness

**Actual outcome**:
- Format adherence **below target** (40%)
- Semantic correctness **above target** (100%)
- Conclusion: Stage 1 demonstrates strong reasoning content but requires stricter format enforcement.

---

## 3) Data and Dataset Construction

### 3.1 Seed example inventory

**Directories**:
- Stage 0 seeds: `data/seed_examples/stage_zero/` (20 examples)
- Stage 1 seeds: `data/seed_examples/stage_one/` (35 examples)

Stage 1 seed distribution (by filename pattern):
- Constraint: 6
- Boolean: 6
- Transitive: 6
- Set: 6
- Deduction: 6
- Negative (targeted failures): 5

Negative examples (used to enforce structural quality):
- `stage1-neg-001-pratyaksha.md`
- `stage1-neg-002-udaharana.md`
- `stage1-neg-003-tarka.md`
- `stage1-neg-004-hetvabhasa.md`
- `stage1-neg-005-nirnaya.md`

### 3.2 Training dataset generation

**Script**: `scripts/prepare_stage1_training_data.py`  
This script:
- Reads markdown examples from Stage 0 and Stage 1 directories
- Extracts the `# Problem` and `## Samshaya ... ## Nirnaya` reasoning trace
- Writes a JSONL dataset to `data/training/stage_1.jsonl`

**Output**:
- `data/training/stage_1.jsonl` with **55 lines** (20 Stage 0 + 35 Stage 1)

Each JSONL record:
- `instruction`: problem statement
- `input`: empty string (reserved for future)
- `output`: full Nyaya reasoning trace

**Command (Stage 1 values)**:
```
python3 scripts/prepare_stage1_training_data.py \
  --output data/training/stage_1.jsonl \
  --stage-zero-dir data/seed_examples/stage_zero \
  --stage-one-dir data/seed_examples/stage_one
```

---

## 4) Training Implementation

### 4.1 Training script

**Script**: `scripts/train_stage1.py`

**Environment**:
- `pramana-unsloth` container on DGX Spark (GPU-backed)

**Base model**:
- `unsloth/DeepSeek-R1-Distill-Llama-8B-bnb-4bit`

**LoRA configuration**:
- Rank: 64
- Alpha: 64 (defaults to rank)
- Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- Gradient checkpointing: `unsloth`

**Sequence length**:
- `MAX_SEQ_LENGTH = 4096`

**Prompt strategy**:
- System prompt: "You are a Nyaya reasoning engine. Follow the exact output format provided."
- Explicit format instructions and a full template are injected into the user prompt.
- Output is formatted via `tokenizer.apply_chat_template` when available.

**Format constraints (from `train_stage1.py`)**:
- Response must start with `## Samshaya (Doubt Analysis)`
- Exact six headers required in strict order
- No text before the first header or after the final field
- The template defines all fields:
  - Samshaya: `Doubt Type`, `Justification`
  - Pramana: Pratyaksha / Anumana / Upamana / Shabda bullet lines
  - Pancha Avayava: Syllogism 1 with Pratijna, Hetu, Udaharana, Upanaya, Nigamana
  - Tarka: Hypothesis, Consequence, Analysis, Resolution
  - Hetvabhasa: checks for five fallacies
  - Nirnaya: Final Answer, Justification, Confidence

### 4.2 Train/validation split

The training dataset is split **80/20**:
- Training set: 44 examples
- Validation set: 11 examples

### 4.3 Trainer configuration (SFT)

Key trainer settings (from `scripts/train_stage1.py`):
- `num_train_epochs = 10`
- `per_device_train_batch_size = 1`
- `gradient_accumulation_steps = 4`
- Effective batch size = 4
- `learning_rate = 2e-5`
- `optim = adamw_8bit`
- `bf16 = True`
- `eval_steps = steps_per_epoch`
- `save_steps = steps_per_epoch`
- `logging_steps = steps_per_epoch // 2`
- `load_best_model_at_end = True`
- `metric_for_best_model = eval_loss`

**Training command (defaults)**:
```
python3 scripts/train_stage1.py
```

**Common overrides**:
```
MAX_SEQ_LENGTH=4096 NUM_TRAIN_EPOCHS=10 LORA_RANK=64 \
MODEL_NAME=unsloth/DeepSeek-R1-Distill-Llama-8B-bnb-4bit \
OUTPUT_DIR=models/stage_1 \
python3 scripts/train_stage1.py
```

### 4.4 Training observability

**Callback**: `NyayaMetricsCallback` (`src/pramana/application/training/callbacks.py`)

Metrics computed per evaluation step:
- `format_adherence`
- `phase_count`
- `syllogism_count`

If `report_to` includes `wandb`, metrics and sample generations are logged to W&B.

---

## 5) Evaluation Results

**File**: `results/stage_1_evaluation.json`  
**Evaluation set**: 10 examples  
- `pramana-003`, `pramana-005`
- `test-001` to `test-008`

### 5.1 Summary metrics

- **Format adherence**: 0.40 (4/10), 95% CI [0.168, 0.687]
- **Semantic answer correctness**: 1.00 (10/10), 95% CI [0.510, 1.0]

**Evaluation command (Stage 1 values)**:
```
MODEL_DIR=models/stage_1 \
VALIDATION_DIR=data/validation/stage_zero \
RESULTS_FILE=results/stage_1_evaluation.json \
python3 scripts/evaluate_stage0.py
```

Evaluation tiers are controlled by `EVAL_TIERS` and include:
- Tier 1 structural validation (always enabled in results)
- Tier 3 Z3 verification when examples are marked verifiable (skipped otherwise)

### 5.2 Parsing outcomes

**Parse success**: 4/10  
**Parse failures**: 6/10

Successful format examples:
- `test-001`
- `test-006`
- `test-007`
- `test-008`

Parse failure modes (counts):
- Missing required section: `Hetvabhasa` (2)
- Missing required section: `Nirnaya` (1)
- Missing required field: `Justification` (1)
- Invalid doubt type: `vipratipatti_samshaya` (1)
- Invalid doubt type: `pramana_dharma` (1)

### 5.3 Interpretation

Stage 1 reliably answers problems correctly but often violates the strict Nyaya output schema. This indicates:
- The model learned the **content** but not the **strict structure**
- Format instruction strength and/or validation must be tightened in Stage 2

### 5.4 Representative Examples and Cross-Stage Ablation

Representative example tables (base vs tuned) and cross-stage tuned comparisons:
- Stage 0 (3 examples): `docs/figures_examples_v1/stage0_representative_examples.csv`
- Stage 1 (3 examples): `docs/figures_examples_v1/stage1_representative_examples.csv`
- Cross-stage tuned vs tuned: `docs/figures_examples_v1/cross_stage_representative_examples.csv`

Cross-stage ablation results (format prompting × decoding temperature):
- Stage 0 summary: `docs/figures_ablation_v1/stage0_ablation_summary.csv`
- Stage 1 summary: `docs/figures_ablation_v1/stage1_ablation_summary.csv`

---

## 6) Model Artifacts and Publishing

### 6.1 Adapter (LoRA)

**HF repo**: `qbz506/nyaya-deepseek-8b-stage1`  
This contains the LoRA adapter weights and tokenizer files for Stage 1.

### 6.2 Full merged model

**HF repo**: `qbz506/nyaya-deepseek-8b-stage1-full`  
This contains the merged model (base + LoRA) saved via `PeftModel.merge_and_unload()`.

### 6.3 Dataset (Stage 1)

**HF dataset**: `qbz506/pramana-nyaya-stage1`  
Includes `train.jsonl` derived from `data/training/stage_1.jsonl` (55 examples).

### 6.4 Local merge artifact (container)

**Local path (pramana-unsloth container)**:
- `/workspace/pramana/hf_upload_full/nyaya-deepseek-8b-stage1-merged`

This directory holds the merged model ready for:
- HF push (full weights)
- GGUF conversion (Ollama)

### 6.5 Merge command (Stage 1 values)

Executed in `pramana-unsloth`:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_id = "unsloth/DeepSeek-R1-Distill-Llama-8B"
adapter_path = "models/stage_1"
out_dir = "/workspace/pramana/hf_upload_full/nyaya-deepseek-8b-stage1-merged"

tokenizer = AutoTokenizer.from_pretrained(base_id, use_fast=True)
base = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype="auto")
merged = PeftModel.from_pretrained(base, adapter_path)
merged = merged.merge_and_unload()
merged.save_pretrained(out_dir, safe_serialization=True)
tokenizer.save_pretrained(out_dir)
```

---

## 7) GGUF Conversion and Ollama Integration

### 7.1 Conversion steps (pramana-unsloth)

**Tooling**: `llama.cpp`

Key steps:
1) Install `sentencepiece` (required for Llama 3 tokenizers)
2) Build `llama.cpp` with CMake
3) Convert HF model to F16 GGUF with `convert_hf_to_gguf.py`
4) Quantize to Q4_K_M with `llama-quantize`

### 7.2 Ollama import (open-webui)

Artifacts are copied into the `open-webui` container and imported into Ollama via a `Modelfile.q4`:

```
FROM /opt/nyaya-deepseek-8b-stage1-merged
SYSTEM "You are a Nyaya reasoning engine. Follow the exact output format provided."
PARAMETER temperature 0
PARAMETER top_p 1
PARAMETER num_ctx 1024
```

Create the model with (example name):
```
ollama create nyaya-deepseek-8b-stage1-q4 -f Modelfile.q4
```

Smoke test:
```
ollama run nyaya-deepseek-8b-stage1-q4 "Solve the problem using Nyaya structure."
```

---

## 8) Hugging Face Space Deployment

**Space**: `qbz506/pramana-nyaya-demo`  
**App file**: `spaces/pramana-nyaya-demo/app.py`

### 8.1 Stage selector integration

The demo now supports **Stage 0** and **Stage 1** via a radio selector.  
Core changes in `app.py`:
- `STAGE_CONFIG` dictionary for stage-specific settings:
  - Base model ID
  - Tuned model ID
  - Dataset repo ID
  - System prompt
  - Default prompt
  - `max_new_tokens`
  - `cache_models`
- Stage info summary block and dynamic updates on selector change
- Example dropdown populated from stage-specific dataset

Stage 1 runtime settings:
- `max_new_tokens = 256`
- `cache_models = false`

Stage 0 runtime settings:
- `max_new_tokens = 512` (default)
- `cache_models = true`

Dataset loading details:
- Uses `hf_hub_download` on the dataset repo ID
- Reads `TRAINING_FILE` (default `train.jsonl`)
- Requires a Space secret `HF_TOKEN` for authenticated downloads

### 8.2 Base vs tuned model comparison

The app continues to show **side-by-side outputs**:
- Base model output
- Tuned model output

For Stage 1, the base model is:
`unsloth/DeepSeek-R1-Distill-Llama-8B`

For Stage 1, the tuned model is:
`qbz506/nyaya-deepseek-8b-stage1-full`

### 8.3 ZeroGPU runtime mitigation

Initial Stage 1 runs hit `GPU task aborted` errors on ZeroGPU.  
Root cause: loading and generating with two 8B models inside a single `@spaces.GPU` task exceeded time/memory limits.

Fixes applied:
- Split inference into two separate GPU tasks:
  - `generate_base(...)`
  - `generate_tuned(...)`
- Chain tasks with `.then()` to keep UI flow
- Reduce Stage 1 `max_new_tokens` to **256**
- Disable caching for Stage 1 models (`cache_models = false`) to free memory per run

### 8.4 Tokenizer compatibility fallback

HF Space runtime raised:
```
ValueError: Tokenizer class TokenizersBackend does not exist or is not currently imported.
```

Fix:
- `AutoTokenizer.from_pretrained(...)` is wrapped in a `try/except`
- On `TokenizersBackend` errors, fallback to `LlamaTokenizerFast.from_pretrained(...)`

### 8.5 Launch behavior

Removed `share=True` from `demo.launch()` to avoid warnings in Spaces.

---

## 9) Space Operations and Verification

### 9.1 Update workflow (open-webui container)

1) Pull the Space repo in `/opt/pramana-nyaya-demo-repo`
2) Edit `app.py` and `README.md`
3) `git add`, `git commit`, `git push`

### 9.2 Logs and restart

Useful endpoints:
- Build logs: `https://huggingface.co/api/spaces/qbz506/pramana-nyaya-demo/logs/build`
- Run logs: `https://huggingface.co/api/spaces/qbz506/pramana-nyaya-demo/logs/run`
- Restart: `POST https://huggingface.co/api/spaces/qbz506/pramana-nyaya-demo/restart`

### 9.3 Runtime checks

The Space exposes:
- `/generate_base`
- `/generate_tuned`

`gradio_client` can be used to run a quick smoke test:
```
from gradio_client import Client
client = Client("qbz506/pramana-nyaya-demo")
client.predict("Stage 1 (DeepSeek 8B)", "Solve the problem using Nyaya structure.", "", api_name="/generate_base")
```

---

## 10) Known Issues and Fixes (Stage 1)

**Issue**: TokenizersBackend error (Llama 3 tokenizer)  
**Fix**: fallback to `LlamaTokenizerFast` in `app.py`

**Issue**: ZeroGPU task abort when running base+tuned in one call  
**Fix**: split into two GPU calls + reduce max tokens + disable caching

**Issue**: Format adherence only 40%  
**Status**: Outstanding. Requires format reinforcement in Stage 2.

---

## 11) Current Artifacts and Key Files

**Training**:
- `scripts/train_stage1.py`
- `data/training/stage_1.jsonl`
- `data/seed_examples/stage_one/`

**Evaluation**:
- `results/stage_1_evaluation.json`

**HF publish**:
- `qbz506/nyaya-deepseek-8b-stage1`
- `qbz506/nyaya-deepseek-8b-stage1-full`
- `qbz506/pramana-nyaya-stage1`

**Space**:
- `spaces/pramana-nyaya-demo/app.py`
- `spaces/pramana-nyaya-demo/README.md`

---

## 12) Lessons Learned and Stage 2 Implications

1) **Format enforcement must be stronger than content learning**.  
   Stage 1 answers are correct but often fail strict schema parsing.

2) **ZeroGPU constraints require split workloads**.  
   Base and tuned comparisons must be separate GPU tasks, especially for 8B models.

3) **Tokenizer compatibility matters for Llama 3 family**.  
   A robust fallback is required in production.

4) **Small datasets can produce strong content correctness**  
   but do not guarantee format discipline. Stage 2 must expand data and reinforce output structure.

---

## 13) Next Steps (Stage 2 Readiness)

- Increase structural penalties or add parser-based filtering in data generation
- Add explicit format verification in training (reject or re-write invalid outputs)
- Expand dataset beyond 55 samples
- Add Z3-backed verification for logic subsets
- Continue Space runtime monitoring with ZeroGPU constraints in mind

---

## 14) Paper Appendix (Numerical + Plots)

Comprehensive numeric inputs/outputs and plots are in:
- `docs/stage_1_paper_appendix.md`
- `docs/figures_stage1_v2/stage1_train_loss.csv`
- `docs/figures_stage1_v2/stage1_eval_loss.csv`
- `docs/figures_stage1_v2/stage1_train_loss.png`
- `docs/figures_stage1_v2/stage1_eval_loss.png`
- `docs/figures_stage1_v2/stage1_train_eval_overlay_step.png`
- `docs/figures_stage1_v2/stage1_train_eval_overlay_epoch.png`
- `docs/figures_stage1_v2/stage1_train_loss_epoch.png`
- `docs/figures_stage1_v2/stage1_eval_loss_epoch.png`
- `docs/figures_stage1_v2/stage1_loss_summary.csv`
- `docs/figures_stage1_v2/stage1_loss_summary.tex`
- `docs/figures_stage1_v2/stage1_parse_error_breakdown.png`
- `docs/figures_stage1_v2/stage1_parse_error_breakdown.csv`
- `docs/figures_stage1_v2/stage1_parse_error_breakdown.tex`
- `docs/figures_stage1_v2/stage1_eval_summary.tex`
- `docs/figures_stage1_v2/stage1_base_vs_tuned_metrics.csv`
- `docs/figures_stage1_v2/stage1_base_vs_tuned_metrics.json`
- `docs/figures_stage1_v2/stage1_base_vs_tuned_metrics.tex`
- `docs/figures_stage1_v2/stage1_base_vs_tuned_metrics.png`
- `docs/figures_stage0_v2/stage0_train_loss.csv`
- `docs/figures_stage0_v2/stage0_eval_loss.csv`
- `docs/figures_stage0_v2/stage0_train_loss.png`
- `docs/figures_stage0_v2/stage0_eval_loss.png`
- `docs/figures_stage0_v2/stage0_train_eval_overlay_step.png`
- `docs/figures_stage0_v2/stage0_train_eval_overlay_epoch.png`
- `docs/figures_stage0_v2/stage0_train_loss_epoch.png`
- `docs/figures_stage0_v2/stage0_eval_loss_epoch.png`
- `docs/figures_stage0_v2/stage0_loss_summary.csv`
- `docs/figures_stage0_v2/stage0_loss_summary.tex`
- `docs/figures_stage0_v2/stage0_parse_error_breakdown.png`
- `docs/figures_stage0_v2/stage0_parse_error_breakdown.csv`
- `docs/figures_stage0_v2/stage0_parse_error_breakdown.tex`
- `docs/figures_stage0_v2/stage0_eval_summary.tex`
- `docs/figures_stage0_v2/stage0_output_length_hist.png`
- `docs/figures_stage0_v2/stage0_base_vs_tuned_metrics.csv`
- `docs/figures_stage0_v2/stage0_base_vs_tuned_metrics.json`
- `docs/figures_stage0_v2/stage0_base_vs_tuned_metrics.tex`
- `docs/figures_stage0_v2/stage0_base_vs_tuned_metrics.png`
- `docs/figures_combined_v1/stage_combined_metrics.csv`
- `docs/figures_combined_v1/stage_combined_metrics.tex`
- `docs/figures_combined_v1/stage_combined_metrics.png`
