# Vyapti Probe Benchmark — Evaluation Report

## Overview

| Model | Accuracy | Probe Acc | Control Acc |
|-------|----------|-----------|-------------|
| llama_3b_base | 50.0% | 38.0% | 62.0% |
| deepseek_8b_base | 62.0% | 46.0% | 78.0% |
| stage0_pramana | 45.0% | 20.0% | 70.0% |
| stage1_pramana | 71.0% | 66.0% | 76.0% |
| base_with_cot | 50.0% | 36.0% | 64.0% |
| base_with_nyaya_template | 64.0% | 52.0% | 76.0% |

## Key Statistical Comparisons

### C1: Probe vs Control (Base Models)

Do entropy-minimizing models fail more on vyapti-requiring problems?

- Difference: +0.280 (95% CI: [+0.140, +0.420])
- **Significant** (p ≈ 0.0000)
- N = 100

### C2: Pramana vs Base DeepSeek (Probes)

Does Nyaya training improve performance on vyapti-requiring problems?

- Difference: +0.200 (95% CI: [+0.020, +0.380])
- **Significant** (p ≈ 0.0184)
- N = 50

### C3: Fine-tuned Stage 1 vs Prompted Template (Probes)

Does fine-tuning provide advantage over just prompting with Nyaya template?

- Difference: +0.140 (95% CI: [-0.040, +0.320])
- Not significant (p ≈ 0.0761)
- N = 50

### C4: Hetvabhasa Taxonomy Coverage

What percentage of failures map to exactly one Hetvabhasa category?

- Difference: +1.000 (95% CI: [+0.000, +0.000])
- **Significant** (p ≈ 0.0000)
- N = 258

## Hetvabhasa Taxonomy Coverage

- Total failures across all models: 258
- Classified into Hetvabhasa categories: 258 (100.0%)
- Unclassified: 0
- Predictive accuracy (classified type matches ground truth): 64.7%

### Distribution

| Category | Count |
|----------|-------|
| savyabhichara | 174 |
| sadhyasama | 31 |
| viruddha | 28 |
| prakaranasama | 13 |
| kalatita | 12 |

## Category-wise Performance

### llama_3b_base

| Category | Probe | Control |
|----------|-------|---------|
| savyabhichara | 4/15 | 7/15 |
| viruddha | 5/10 | 6/10 |
| prakaranasama | 3/10 | 5/10 |
| sadhyasama | 6/10 | 8/10 |
| kalatita | 1/5 | 5/5 |

### deepseek_8b_base

| Category | Probe | Control |
|----------|-------|---------|
| savyabhichara | 6/15 | 11/15 |
| viruddha | 7/10 | 6/10 |
| prakaranasama | 6/10 | 8/10 |
| sadhyasama | 3/10 | 9/10 |
| kalatita | 1/5 | 5/5 |

### stage0_pramana

| Category | Probe | Control |
|----------|-------|---------|
| savyabhichara | 3/15 | 11/15 |
| viruddha | 2/10 | 7/10 |
| prakaranasama | 0/10 | 6/10 |
| sadhyasama | 3/10 | 7/10 |
| kalatita | 2/5 | 4/5 |

### stage1_pramana

| Category | Probe | Control |
|----------|-------|---------|
| savyabhichara | 8/15 | 14/15 |
| viruddha | 9/10 | 7/10 |
| prakaranasama | 7/10 | 6/10 |
| sadhyasama | 6/10 | 6/10 |
| kalatita | 3/5 | 5/5 |

### base_with_cot

| Category | Probe | Control |
|----------|-------|---------|
| savyabhichara | 4/15 | 11/15 |
| viruddha | 4/10 | 6/10 |
| prakaranasama | 3/10 | 7/10 |
| sadhyasama | 6/10 | 5/10 |
| kalatita | 1/5 | 3/5 |

### base_with_nyaya_template

| Category | Probe | Control |
|----------|-------|---------|
| savyabhichara | 7/15 | 11/15 |
| viruddha | 5/10 | 8/10 |
| prakaranasama | 6/10 | 6/10 |
| sadhyasama | 5/10 | 8/10 |
| kalatita | 3/5 | 5/5 |
