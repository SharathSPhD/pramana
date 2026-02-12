# Vyapti Probe Benchmark — Evaluation Report

## Overview

| Model | Accuracy | Probe Acc | Control Acc |
|-------|----------|-----------|-------------|
| deepseek_8b_base | 32.0% | 28.0% | 36.0% |
| base_with_cot | 36.0% | 36.0% | 36.0% |
| base_with_nyaya_template | 25.0% | 22.0% | 28.0% |
| stage1_pramana | 27.0% | 24.0% | 30.0% |
| llama_3b_base | 39.0% | 42.0% | 36.0% |
| stage0_pramana | 25.0% | 20.0% | 30.0% |

## Key Statistical Comparisons

### C1: Probe vs Control (Base Models)

Do entropy-minimizing models fail more on vyapti-requiring problems?

- Difference: +0.010 (95% CI: [-0.100, +0.120])
- Not significant (p ≈ 0.9258)
- N = 100

### C2: Pramana vs Base DeepSeek (Probes)

Does Nyaya training improve performance on vyapti-requiring problems?

- Difference: -0.040 (95% CI: [-0.120, +0.040])
- Not significant (p ≈ 0.4430)
- N = 50

### C3: Fine-tuned Stage 1 vs Prompted Template (Probes)

Does fine-tuning provide advantage over just prompting with Nyaya template?

- Difference: +0.020 (95% CI: [-0.040, +0.080])
- Not significant (p ≈ 0.7782)
- N = 50

### C4: Hetvabhasa Taxonomy Coverage (Descriptive)

What percentage of failures map to exactly one Hetvabhasa category?

- Coverage: 1.000
- Descriptive metric (no inferential test)
- N = 416

## Hetvabhasa Taxonomy Coverage

- Total failures across all models: 416
- Classified into Hetvabhasa categories: 416 (100.0%)
- Unclassified: 0
- Assisted predictive accuracy (includes fallback): 77.4%
- Strict predictive accuracy (excludes fallback): 66.7%
- Fallback classifications: 134

### Distribution

| Category | Count |
|----------|-------|
| savyabhichara | 188 |
| sadhyasama | 79 |
| viruddha | 74 |
| kalatita | 57 |
| prakaranasama | 18 |

## Category-wise Performance

### deepseek_8b_base

| Category | Probe | Control |
|----------|-------|---------|
| savyabhichara | 0/15 | 0/15 |
| viruddha | 5/10 | 4/10 |
| prakaranasama | 6/10 | 9/10 |
| sadhyasama | 1/10 | 3/10 |
| kalatita | 2/5 | 2/5 |

### base_with_cot

| Category | Probe | Control |
|----------|-------|---------|
| savyabhichara | 1/15 | 0/15 |
| viruddha | 6/10 | 4/10 |
| prakaranasama | 7/10 | 8/10 |
| sadhyasama | 1/10 | 4/10 |
| kalatita | 3/5 | 2/5 |

### base_with_nyaya_template

| Category | Probe | Control |
|----------|-------|---------|
| savyabhichara | 1/15 | 0/15 |
| viruddha | 4/10 | 2/10 |
| prakaranasama | 5/10 | 8/10 |
| sadhyasama | 0/10 | 3/10 |
| kalatita | 1/5 | 1/5 |

### stage1_pramana

| Category | Probe | Control |
|----------|-------|---------|
| savyabhichara | 1/15 | 1/15 |
| viruddha | 4/10 | 2/10 |
| prakaranasama | 6/10 | 8/10 |
| sadhyasama | 0/10 | 3/10 |
| kalatita | 1/5 | 1/5 |

### llama_3b_base

| Category | Probe | Control |
|----------|-------|---------|
| savyabhichara | 8/15 | 1/15 |
| viruddha | 4/10 | 4/10 |
| prakaranasama | 5/10 | 8/10 |
| sadhyasama | 2/10 | 4/10 |
| kalatita | 2/5 | 1/5 |

### stage0_pramana

| Category | Probe | Control |
|----------|-------|---------|
| savyabhichara | 0/15 | 1/15 |
| viruddha | 4/10 | 2/10 |
| prakaranasama | 5/10 | 8/10 |
| sadhyasama | 0/10 | 3/10 |
| kalatita | 1/5 | 1/5 |
