# Pramana Implementation Report (Stage 0)

Date: 2026-01-31  
Scope: Stage 0 proof-of-concept implementation and training on DGX Spark

## Executive Summary

This repository implements a full Stage 0 pipeline for Nyaya-structured reasoning: data parsing, validation, training orchestration, evaluation scaffolding, and a Unsloth-based fine-tuning run on DGX Spark. The training run completed successfully and produced LoRA adapter artifacts, but evaluation/validation against held-out examples was not executed, so success criteria for format generalization remain unverified.

## Implementation Overview

**Architecture**
- **Domain**: Nyaya models, reward components, structure validator.
- **Application**: Markdown parsing, evaluation pipeline, training orchestration.
- **Infrastructure**: Unsloth adapter, checkpoint repository, Z3 verification, HF upload.
- **CLI**: `train`, `evaluate`, `validate`, `data` commands.

**Key Components (selected)**
- `src/pramana/domain/models/nyaya_example.py`: 6-phase Nyaya schema.
- `src/pramana/domain/validators/structure.py`: phase/Pramana/syllogism validation.
- `src/pramana/application/evaluation/pipeline.py`: 3-tier evaluation chain.
- `src/pramana/infrastructure/verification/z3_verifier.py`: SMT-LIB validation.
- `scripts/prepare_training_data.py`: Markdown → JSONL conversion.
- `scripts/train_unsloth_dgx.py`: Unsloth fine-tuning on DGX Spark.

## Seed Examples (Stage 0)

All seed examples are under `data/seed_examples/stage_zero/` and are marked `verified: true` with `z3_verifiable: true` in frontmatter.

- **pramana-001 (constraint_satisfaction)**: classic assignment puzzle (pet ownership).
- **pramana-002 (boolean_sat)**: Knights/Knaves logical consistency.
- **pramana-003 (transitive_reasoning)**: total ordering from pairwise comparisons.
- **pramana-004 (set_membership)**: partitioning into two groups with constraints.
- **pramana-005 (multi_step_deduction)**: chained implication (P → Q → R → S).

## Training Configuration (Stage 0)

**Script**: `scripts/train_unsloth_dgx.py`  
**Dataset**: `data/training/stage_0.jsonl` (5 examples)

**Model**
- Base model: `unsloth/Llama-3.2-3B-Instruct-bnb-4bit`
- Quantization: 4-bit
- LoRA rank/alpha: 32 / 32
- Trainable parameters: 48,627,712 (~1.49% of 3.26B)
- Max sequence length: 2048
- Mixed precision: `bf16=True`, `fp16=False`

**Training**
- Batch size: 1 (gradient accumulation: 4; total batch size: 4)
- Steps: 50
- Epochs: 25
- Warmup steps: 2
- Optimizer: `adamw_8bit`
- LR: 2e-5
- Checkpoints: every 25 steps

## Training Metrics (Latest Run)

**Source**: DGX Spark Unsloth log  
**Run Summary**
- Train runtime: **123.6s**
- Train loss (final): **0.9898**
- Samples/sec: **1.619**
- Steps/sec: **0.405**

**Loss Trend (selected)**
- Early: ~**1.225** @ epoch 1
- Mid: ~**0.895** @ epoch 9
- Late: ~**0.860** @ epoch 25

## Tuned vs Untuned Model

**Untuned (Base)**
- Source: HF/Unsloth base weights
- Parameters: Full 3.26B base model
- Behavior: general-purpose instruction tuning

**Tuned (Stage 0)**
- Artifacts: `models/stage_0/adapter_model.safetensors`, `adapter_config.json`
- Scope: LoRA adapters only (no full model merge)
- Loading: Requires base model + adapter
- Intended behavior: structured Nyaya 6-phase responses

## Verification and Validation

**Structural Validation**
- `NyayaStructureValidator` checks:
  - 6-phase completeness
  - presence of at least one Pramana source
  - completeness of 5-member syllogisms

**Evaluation Pipeline (3 tiers)**
- Tier 1: structural validation handler
- Tier 2: LLM judge rubric for quality and adherence
- Tier 3: Z3 verification (SMT-LIB constraints)

**Z3 Verification**
- `Z3Verifier` parses SMT-LIB, checks SAT/UNSAT, extracts model.

**Gap**: Stage 0 training did **not** run any of these validation stages post-training.

## Critical Review of Training Output

**Major Risks**
1. **No evaluation**: no held-out dataset or post-training validation to check Nyaya format adherence.
2. **Overfitting risk**: 25 epochs on 5 examples with no validation split.
3. **Hyperparameter mismatch vs project guidelines**: LoRA rank 32 and seq length 2048 below recommended 64–128 / 4096+.

**Warnings Observed**
- Batch size 1 reduces padding-free training benefits.
- `num_proc` auto-reduced due to tiny dataset.

## Recommendations (Next Steps)

**Immediate**
- Create a held-out test set (1–2 examples).
- Run the evaluation pipeline on unseen prompts.
- Log format adherence metrics (phase completeness, ordering).

**Before Stage 1**
- Align hyperparameters with project guidance (LoRA rank 64–128, seq length 4096+).
- Add experiment tracking (W&B/TensorBoard).
- Expand seed set beyond 5 examples for more robust generalization checks.

## Artifacts Produced

**LoRA adapter outputs**: `models/stage_0/`  
Includes: `adapter_model.safetensors`, `adapter_config.json`, tokenizer artifacts, checkpoints.

## Appendix: Key Paths

- Seed examples: `data/seed_examples/stage_zero/`
- Training dataset: `data/training/stage_0.jsonl`
- Training script: `scripts/train_unsloth_dgx.py`
- Evaluation pipeline: `src/pramana/application/evaluation/`
- Structural validation: `src/pramana/domain/validators/structure.py`
- Z3 verification: `src/pramana/infrastructure/verification/z3_verifier.py`
