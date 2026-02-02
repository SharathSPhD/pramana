---
id: stage1-019
problem_type: set_membership
difficulty: easy
variables: 3
ground_truth: "Basket A has the banana, Basket B has the apple, Basket C has the cherry"
metadata:
  created_date: 2026-02-02
  author: stage1
  validated: false
  z3_verifiable: false
  stage: 1
---

# Problem

Three baskets (A, B, C) each contain one fruit: apple, banana, or cherry.

**Constraints**:
1. Basket B has the apple.
2. Basket A does not have the cherry.
3. Basket C does not have the banana.

**Question**: Which fruit is in each basket?

---

## Samshaya (Doubt Analysis)

**Doubt Type**: Samana Dharma Upapatti (Multiple possibilities share similar properties)

**Justification**: The fruit placement is uncertain until exclusions fix each basket.

---

## Pramana (Sources of Knowledge)

### Pratyaksha (Direct Perception)
- Basket B has the apple.
- Basket A does not have the cherry.
- Basket C does not have the banana.
- Each basket has exactly one fruit.

### Anumana (Inference)
- Since B has apple, A and C cannot have apple.
- C cannot have banana, so C must have cherry.
- A then must have banana.

### Upamana (Comparison)
- This is a standard one-to-one assignment in sets.

### Shabda (Testimony)
- If an item is fixed, remaining items distribute to remaining slots.

---

## Pancha Avayava (5-Member Syllogism)

### Syllogism 1: Determine Basket C
**Pratijna (Thesis)**: Basket C has the cherry.  
**Hetu (Reason)**: Basket B has the apple and Basket C cannot have banana.  
**Udaharana (Universal + Example)**: Wherever only one option remains after exclusions, the remaining option must hold.  
**Upanaya (Application)**: C cannot have apple or banana, so cherry remains.  
**Nigamana (Conclusion)**: Therefore, Basket C has the cherry.

---

## Tarka (Counterfactual Reasoning)

**Hypothesis**: Suppose Basket C does not have the cherry.  
**Consequence**: C would need apple or banana, but apple is in B and banana is forbidden.  
**Analysis**: This contradicts the constraints.  
**Resolution**: Therefore, Basket C must have the cherry.

---

## Hetvabhasa (Fallacy Check)

Check for Savyabhichara: No  
Check for Viruddha: No  
Check for Asiddha: No  
Check for Satpratipaksha: No  
Check for Badhita: No

---

## Nirnaya (Ascertainment)

**Final Answer**: Basket A has the banana, Basket B has the apple, Basket C has the cherry.  
**Justification**: B is fixed to apple, C cannot take banana, so C takes cherry and A takes banana.  
**Confidence**: High
