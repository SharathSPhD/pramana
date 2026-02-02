---
id: stage1-neg-001
problem_type: constraint_satisfaction
difficulty: easy
variables: 3
ground_truth: "Asha has the cat, Bo has the dog, Cy has the fish"
negative_example: true
metadata:
  created_date: 2026-02-02
  author: stage1
  validated: false
  z3_verifiable: false
  stage: 1
---

# Problem

Three friends (Asha, Bo, Cy) each have one pet: cat, dog, or fish.

**Constraints**:
1. Bo has the dog.
2. Asha does not have the fish.
3. Cy does not have the dog.

**Question**: Who has which pet?

---

## Samshaya (Doubt Analysis)

**Doubt Type**: Samana Dharma Upapatti (Multiple possibilities share similar properties)

**Justification**: The assignment is unclear until exclusions are applied.

---

## Pramana (Sources of Knowledge)

### Pratyaksha (Direct Perception)
- Bo has the dog.
- Asha does not have the fish.
- Cy does not have the dog.
- **Therefore Asha has the cat.**  <!-- Negative: inferred fact in Pratyaksha -->

### Anumana (Inference)
- Since Bo has the dog, Asha and Cy cannot have the dog.
- Asha cannot have the fish, so she must have the cat.
- Cy then has the fish.

### Upamana (Comparison)
- This follows typical assignment elimination.

### Shabda (Testimony)
- If one item is fixed, remaining items distribute to remaining people.

---

## Pancha Avayava (5-Member Syllogism)

### Syllogism 1: Determine Asha's Pet
**Pratijna (Thesis)**: Asha has the cat.  
**Hetu (Reason)**: Bo has the dog and Asha does not have the fish.  
**Udaharana (Universal + Example)**: Wherever only one option remains after exclusions, the remaining option must hold.  
**Upanaya (Application)**: Asha cannot have dog or fish, so cat remains.  
**Nigamana (Conclusion)**: Therefore, Asha has the cat.

---

## Tarka (Counterfactual Reasoning)

**Hypothesis**: Suppose Asha does not have the cat.  
**Consequence**: She would need dog or fish, both excluded.  
**Analysis**: This is a contradiction with the constraints.  
**Resolution**: Therefore, Asha must have the cat.

---

## Hetvabhasa (Fallacy Check)

Check for Savyabhichara: No  
Check for Viruddha: No  
Check for Asiddha: No  
Check for Satpratipaksha: No  
Check for Badhita: No

---

## Nirnaya (Ascertainment)

**Final Answer**: Asha has the cat, Bo has the dog, Cy has the fish.  
**Justification**: Bo is fixed to dog, Asha cannot take fish, leaving cat for Asha and fish for Cy.  
**Confidence**: High
