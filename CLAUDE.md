# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Pramana** is a research project building an epistemic reasoning engine for AI systems based on Navya-Nyaya logic. The goal is to create LLMs that apply 2,500-year-old Indian epistemological methodology to solve logical problems systematically, rather than relying on probabilistic pattern-matching.

### Core Concept

Unlike standard chain-of-thought reasoning, Pramana enforces a structured 6-phase Nyaya methodology:

1. **Samshaya** (Doubt Analysis) - Classify the type of uncertainty/ambiguity
2. **Pramana** (Evidence Sources) - Identify valid knowledge sources (Pratyaksha/perception, Anumana/inference, Upamana/comparison, Shabda/testimony)
3. **Pancha Avayava** (5-Member Syllogism) - Construct formal argument with Pratijna (thesis), Hetu (reason), Udaharana (universal example), Upanaya (application), Nigamana (conclusion)
4. **Tarka** (Counterfactual Testing) - Use reductio ad absurdum to verify conclusions
5. **Hetvabhasa** (Fallacy Detection) - Check for reasoning errors (Savyabhichara/erratic, Viruddha/contradictory, etc.)
6. **Nirnaya** (Ascertainment) - Reach definitive conclusion or explicitly state insufficient evidence

## Implementation Approach

### Data Generation Strategy

Training data consists of logical problems (constraint satisfaction, Boolean SAT, multi-step deduction) paired with complete Nyaya-structured reasoning traces. The approach uses:

- **Manual seed creation**: 50-100 gold-standard examples demonstrating perfect Nyaya methodology
- **Synthetic expansion**: Use frontier models (GPT-4o, Claude) to generate examples following seed patterns
- **Quality filtering**: Statistical sampling + Z3 SMT solver verification for formal logic problems

### Training Pipeline

1. **Base model selection**: DeepSeek-R1-Distill-Llama-8B (has pre-trained reasoning traces) or Qwen 2.5-14B (strong logic capabilities)
2. **Supervised Fine-Tuning**: Use Unsloth with QLoRA (4-bit quantization), high LoRA rank (64-128) targeting all attention/FFN layers
3. **Reinforcement Learning**: Apply GRPO (Group Relative Policy Optimization) with composite reward function:
   - Format reward (structural adherence to 6 phases)
   - Validity reward (Process Reward Model + ground truth matching)
   - Consistency reward (Tarka counterfactual verification via Z3)

### Neuro-Symbolic Verification

For formal logic problems, implement runtime verification:
- Parse Pratijna/Hetu/Udaharana from model output
- Autoformalize to Z3 SMT-LIB format
- Execute Z3 solver to verify logical validity
- If invalid, inject error feedback and trigger model self-correction

## Development Guidelines

### Data Format

Training examples must follow strict JSON schema with required fields for each Nyaya phase. The Udaharana (example) section must contain explicit "Wherever X, there is Y" universal rule statements grounded in concrete instances.

### Evaluation Metrics

- **Format Adherence Rate**: >95% properly structured 6-phase outputs
- **Logical Validity**: Z3 verification success rate on formal logic subset
- **Reasoning Quality**: Human evaluation of appropriate Pramana usage, meaningful Tarka testing, correct Hetvabhasa classification
- **Benchmark Performance**: Test on LogicBench, ProntoQA, RuleTaker, GSM8K subset

### Critical Constraints

- **No premature feature creep**: Build core reasoning engine first, add multi-agent debate protocols later
- **Quality over quantity**: 500 high-quality verified examples > 5,000 mediocre synthetic examples
- **Staged validation**: Validate each stage (proof-of-concept → supervised fine-tuning → RL → production) before proceeding
- **Epistemic humility**: Model must distinguish Nirnaya (definitive knowledge) from Tarka (reasonable hypothesis requiring verification)

## Staged Implementation Plan

### Stage 0: Proof of Concept (2 weeks, $100)
- Create 5-10 manual seed examples (constraint satisfaction puzzles)
- Fine-tune smallest model (Llama 3.2-3B) to validate format learnability
- Success criteria: Model attempts 6-phase structure on new problems

### Stage 1: Minimum Viable Reasoner (8-10 weeks, $500-1000)
- Create 50 gold-standard examples (constraint satisfaction + Boolean SAT)
- Fine-tune DeepSeek-R1-Distill-Llama-8B with strong LoRA
- Success criteria: >90% format adherence, 60-70% accuracy on held-out problems

### Stage 2: Synthetic Scaling (8 weeks, $2000-5000)
- Generate 200-500 examples via GPT-4o/Claude with statistical quality control
- Implement Z3 verification for formal logic subset
- Evaluate on standard reasoning benchmarks

### Stage 3: RL Enhancement (8-12 weeks, $10,000-30,000)
- Implement GRPO training with composite reward function
- Train Process Reward Model or use GPT-4 as judge
- Success criteria: High accuracy + systematic reasoning traces

### Stage 4: Production Hardening (timeline variable)
- Multi-agent Vada/Jalpa debate protocols
- Runtime Z3 verification pipeline
- Expand to domains beyond formal logic

## Key Technical Details

### Training Hyperparameters
- Learning rate: 2e-5 (preserve pre-trained reasoning)
- LoRA rank: 64-128 (complex reasoning paradigm requires high capacity)
- Sequence length: 4096+ tokens (full reasoning traces are long)
- Epochs: 10-15 for embedding new reasoning paradigm
- Effective batch size: 16-32 via gradient accumulation

### Compute Requirements
- Stage 0-1: Single A100 40GB, 20-40 GPU-hours
- Stage 2: Single A100, 50-80 GPU-hours
- Stage 3: 4-8 A100s, 672-1344 GPU-hours (2-4 weeks)

### Tooling Stack
- Fine-tuning: Unsloth with QLoRA
- Experiment tracking: Weights & Biases or TensorBoard
- Data management: SQLite + Git version control
- Verification: Z3 SMT Solver for formal logic subset
- Inference: vLLM for optimized serving (later stages)

## Known Risks

1. **Syntactic mimicry without semantic reasoning**: Model generates correct format but logically incoherent content → Mitigation: Z3 verification + extensive human evaluation
2. **Domain overfitting**: Works for logic puzzles but fails on broader reasoning → Mitigation: Early testing on diverse problem types
3. **Synthetic data poisoning**: Scaled generation produces subtle errors → Mitigation: Statistical sampling + Z3 auto-verification
4. **Reasoning overhead**: 6-phase structure too verbose for practical use → Mitigation: Track tokens/problem, consider abbreviated forms for simple cases

## Research Context

This project addresses the "epistemic gap" in LLMs demonstrated by Apple's October 2024 research showing 65% performance degradation when irrelevant context is added. Unlike Western formal logic (divorced from epistemology), Navya-Nyaya integrates logic and epistemology, requiring grounding in concrete examples (dṛṣṭānta) and explicit universal rules (vyapti).

The hypothesis: Teaching LLMs a formal epistemological framework through fine-tuning creates better systematic reasoning than generic chain-of-thought, comparable to frontier models like o1/Claude extended thinking but based on explicit methodology rather than opaque RL.
