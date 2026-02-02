# Stage 0 Hyperparameter Justification (LoRA Rank)

## Summary

Stage 0 used a two-pass approach. The initial run selected LoRA rank `r=32` to minimize GPU memory, reduce iteration time, and validate the Nyaya format-learning hypothesis quickly. Results showed insufficient capacity for reliable structural adherence, so the corrected run increased LoRA rank to `r=64`, aligning with the Stage 1 target range of `64–128` while remaining compute-efficient for a small dataset.

## Rationale for the Initial r=32 Choice

- **Proof-of-concept focus**: Stage 0 emphasized rapid validation of the 6-phase structure, not peak accuracy.
- **Small dataset and short turnaround**: Only 5 seed examples in the initial run; a lower rank reduced overfitting risk and iteration cost.
- **GPU efficiency**: Lower rank reduced VRAM footprint and sped up training during early experimentation.

## Why r=32 Was Insufficient

The initial run failed to reliably produce Nyaya-compliant structure (0% parse success in the first evaluation). The observed failure modes indicated under-capacity for modeling the strict format and reasoning steps.

## Justification for the Corrected r=64 Run

- **Capacity increase without excessive cost**: r=64 doubled adaptation capacity while remaining feasible on a single GPU.
- **Alignment with Stage 1 targets**: The project spec recommends `r=64–128` for stable format adherence; r=64 is the lower bound of that range.
- **Empirical improvement**: The corrected run (r=64) achieved consistent structural adherence with strict prompt templating.

## Stage 1 Implications

Stage 1 will target `r=64–128` depending on model size, dataset scale, and validation performance. The Stage 0 experience indicates that ranks below 64 are insufficient for robust Nyaya formatting and reasoning traces.
