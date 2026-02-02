---
id: stage1-neg-005
problem_type: constraint_satisfaction
difficulty: easy
variables: 3
ground_truth: "Iris has the apple, Jude has the banana, Kora has the orange"
negative_example: true
metadata:
  created_date: 2026-02-02
  author: stage1
  validated: false
  z3_verifiable: false
  stage: 1
---

# Problem

Three people (Iris, Jude, Kora) each hold one fruit: apple, banana, or orange.

**Constraints**:
1. Jude has the banana.
2. Iris does not have the orange.
3. Kora does not have the banana.

**Question**: Who has which fruit?

---

## Samshaya (Doubt Analysis)

**Doubt Type**: Samana Dharma Upapatti (Multiple possibilities share similar properties)

**Justification**: The assignment is uncertain until exclusions determine a unique mapping.

---

## Pramana (Sources of Knowledge)

### Pratyaksha (Direct Perception)
- Jude has the banana.
- Iris does not have the orange.
- Kora does not have the banana.

### Anumana (Inference)
- Since Jude has the banana, Iris and Kora cannot have the banana.
- Iris cannot have orange, so Iris must have apple.
- Kora then must have orange.

### Upamana (Comparison)
- This is a direct exclusive assignment.

### Shabda (Testimony)
- If one item is fixed, remaining items distribute to remaining people.

---

## Pancha Avayava (5-Member Syllogism)

### Syllogism 1: Determine Iris's Fruit
**Pratijna (Thesis)**: Iris has the apple.  
**Hetu (Reason)**: Jude has banana and Iris cannot have orange.  
**Udaharana (Universal + Example)**: Wherever only one option remains after exclusions, the remaining option must hold.  
**Upanaya (Application)**: Iris cannot have banana or orange, so apple remains.  
**Nigamana (Conclusion)**: Therefore, Iris has the apple.

---

## Tarka (Counterfactual Reasoning)

**Hypothesis**: Suppose Iris does not have the apple.  
**Consequence**: Iris would need banana or orange, both excluded.  
**Analysis**: This contradicts the constraints.  
**Resolution**: Therefore, Iris must have the apple.

---

## Hetvabhasa (Fallacy Check)

Check for Savyabhichara: No  
Check for Viruddha: No  
Check for Asiddha: No  
Check for Satpratipaksha: No  
Check for Badhita: No

---

## Nirnaya (Ascertainment)

**Final Answer**: Iris has the apple, Jude has the banana, Kora has the orange.  
**Justification**: This is certainly true beyond any possible doubt.  <!-- Negative: overconfident without epistemic humility -->
**Confidence**: High
