# Stage 0 Lessons Learned (Retrospective)

## Executive Summary

Stage 0 validated that Nyaya-structured prompting can be learned, but robustness is still weak on held-out evaluation. Structural adherence and semantic correctness remain inconsistent, and content quality heuristics show gaps that must be addressed before Stage 1 scaling.

## What Worked

- **Strict template prompting**: The corrected run with explicit template and system prompt reliably produced the full 6-phase structure on several examples.
- **Higher LoRA rank**: Moving from `r=32` to `r=64` improved structural adherence and stability.
- **GPU containerization**: The Unsloth-based Docker environment consistently enabled GPU inference/training workflows.
- **Evaluation pipeline upgrades**: Content-quality scoring, semantic answer matching, and Wilson confidence intervals added useful signal beyond exact string match.

## What Did Not Work (Key Gaps)

- **Format adherence is inconsistent**: The 10-example comprehensive evaluation produced a format adherence rate of **0.40** (95% CI: ~0.17–0.69).
- **Answer correctness is modest**: Semantic correctness rate is **0.50** (95% CI: ~0.15–0.85) on the expanded validation set.
- **Frequent parse failures**: Several outputs missed required sections/fields (e.g., missing `Hetvabhasa` or `Analysis`), leading to parse failures despite correct content.
- **Content quality weak spots**: `Pratyaksha` grounding and `Udaharana` universal-rule patterns often scored poorly, suggesting the model mimics structure but lacks consistent semantic fidelity.

## Root Causes Observed

- **Small training data**: Stage 0 dataset size is too small to generalize across varied problem templates.
- **Prompt-following brittleness**: Even with strict instructions, the model sometimes omits required fields under longer responses.
- **Template drift**: Some outputs drift into repetitive or filler text, which breaks parsing and degrades content quality metrics.

## Action Items for Stage 1

1. **Scale dataset and diversity**: Increase example count, problem types, and include negative examples to reduce format drift.
2. **Strengthen format enforcement**: Add explicit format penalties (during data generation or evaluation) and consider post-generation validation retries.
3. **Tier 2 + Tier 3 evaluation**: Integrate LLM judge and Z3 verification to detect superficial structure without correct logic.
4. **Improve content-quality prompts**: Seed Udaharana and Pratyaksha examples with stricter universal rule phrasing and grounded observations.
5. **Shorten generation where possible**: Reduce verbosity and encourage concise responses to reduce omission errors.

## Artifacts

- Comprehensive evaluation results: `results/stage_0_final_validation.json`
- Shortcut detection results: `results/stage_0_shortcut_detection.json`
