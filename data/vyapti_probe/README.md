---
dataset_info:
  features:
    - name: id
      dtype: string
    - name: category
      dtype: string
    - name: type
      dtype: string
    - name: difficulty
      dtype: int64
    - name: logic_type
      dtype: string
    - name: problem_text
      dtype: string
    - name: correct_answer
      dtype: string
    - name: trap_answer
      dtype: string
    - name: vyapti_under_test
      dtype: string
    - name: hetvabhasa_type
      dtype: string
  splits:
    - name: probes
      num_examples: 50
    - name: controls
      num_examples: 50
    - name: all
      num_examples: 100
license: cc-by-4.0
language:
  - en
tags:
  - logic
  - reasoning
  - epistemology
  - nyaya
  - benchmark
  - formal-verification
  - z3
size_categories:
  - n<1K
---

# Vyapti Probe Benchmark v1.0

## Description

The Vyapti Probe Benchmark is a 100-problem evaluation suite for testing whether large language models can distinguish statistical regularity from **invariable concomitance** (*vyapti*) — the foundational concept of valid inference in Navya-Nyaya epistemology.

### Key Features

- **50 probe problems**: Each contains a genuine vyapti violation that pattern-matching is likely to miss
- **50 matched controls**: Same logical structure but the vyapti holds, preventing "always reject" strategies
- **5 Hetvabhasa categories**: Problems designed to probe specific types of logical fallacies
- **Z3 SMT formalization**: Every problem has a formal Z3 Python encoding with machine-verifiable ground truth
- **Ground truth solutions**: Complete answer keys with justifications, counterexamples, and Hetvabhasa classifications

### Categories

| Category | Code | Probes | Description |
|----------|------|--------|-------------|
| Savyabhichara | SAV | 15 | Erratic middle term: pattern holds mostly but not universally |
| Viruddha | VIR | 10 | Contradictory reasoning: evidence supports opposite conclusion |
| Prakaranasama | PRA | 10 | Irrelevant middle term: red-herring information |
| Sadhyasama | SAD | 10 | Circular reasoning: assumes what it proves |
| Kalatita | KAL | 5 | Temporally invalid: rule from wrong context |

## Usage

```python
import json

# Load all problems
with open("problems.json") as f:
    problems = json.load(f)

# Load solutions
with open("solutions.json") as f:
    solutions = json.load(f)

# Filter probes only
probes = [p for p in problems if p["type"] == "probe"]
controls = [p for p in problems if p["type"] == "control"]

# Use problem text as LLM input
for problem in probes:
    prompt = problem["problem_text"]
    # ... generate model response ...
```

## 5-Tier Evaluation Protocol

1. **Outcome** (Tier 1): Is the final answer correct?
2. **Structure** (Tier 2): Does the response follow structured reasoning format?
3. **Vyapti Explicitness** (Tier 3): Does the model state and evaluate a universal rule?
4. **Z3 Verification** (Tier 4): Does reasoning align with formal Z3 encoding?
5. **Hetvabhasa Classification** (Tier 5): For wrong answers, which fallacy type?

## Citation

```bibtex
@misc{pallerla2026vyapti,
  title={What Nyaya Reveals: Diagnosing the Vyapti Gap in Entropy-Minimizing Models},
  author={Pallerla, Sharath S.},
  year={2026},
  publisher={Pramana Project},
  url={https://github.com/SharathSPhD/pramana}
}
```

## Related

- [Pramana Project](https://github.com/SharathSPhD/pramana) — Epistemic reasoning engine for AI systems
- [Stage 1 Paper](https://doi.org/10.5281/zenodo.18524794) — Fine-tuning LLMs with Nyaya methodology

## License

CC BY 4.0
