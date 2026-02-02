---
id: stage1-002
problem_type: constraint_satisfaction
difficulty: easy
variables: 3
ground_truth: "Eli has the blue card, Faye has the red card, Gus has the green card"
metadata:
  created_date: 2026-02-02
  author: stage1
  validated: false
  z3_verifiable: false
  stage: 1
---

# Problem

Three students (Eli, Faye, Gus) each receive one colored card: red, blue, or green.

**Constraints**:
1. Eli has the blue card.
2. Faye does not have the green card.
3. Gus does not have the blue card.

**Question**: Which color does each student have?

---

## Samshaya (Doubt Analysis)

**Doubt Type**: Samana Dharma Upapatti (Multiple possibilities share similar properties)

**Justification**: The card assignments are uncertain until the constraints eliminate the invalid possibilities.

---

## Pramana (Sources of Knowledge)

### Pratyaksha (Direct Perception)
- Eli has the blue card.
- Faye does not have the green card.
- Gus does not have the blue card.
- Each student has exactly one card.

### Anumana (Inference)
- Since Eli has blue, Faye and Gus cannot have blue.
- Faye cannot have green, so she must have red.
- Gus then must have green.

### Upamana (Comparison)
- This is a standard one-to-one assignment puzzle.

### Shabda (Testimony)
- In a unique assignment, remaining items go to remaining people.

---

## Pancha Avayava (5-Member Syllogism)

### Syllogism 1: Determine Faye's Card
**Pratijna (Thesis)**: Faye has the red card.  
**Hetu (Reason)**: Eli already has blue and Faye cannot have green.  
**Udaharana (Universal + Example)**: Wherever only one option remains after exclusions, the remaining option must hold.  
**Upanaya (Application)**: Faye cannot have blue or green, so red remains.  
**Nigamana (Conclusion)**: Therefore, Faye has the red card.

---

## Tarka (Counterfactual Reasoning)

**Hypothesis**: Suppose Faye does not have the red card.  
**Consequence**: She would need blue or green, but blue is Eli's and green is forbidden.  
**Analysis**: This contradicts the constraints.  
**Resolution**: Therefore, Faye must have the red card.

---

## Hetvabhasa (Fallacy Check)

Check for Savyabhichara: No  
Check for Viruddha: No  
Check for Asiddha: No  
Check for Satpratipaksha: No  
Check for Badhita: No

---

## Nirnaya (Ascertainment)

**Final Answer**: Eli has the blue card, Faye has the red card, Gus has the green card.  
**Justification**: Eli is fixed to blue, Faye cannot be green, leaving red for Faye and green for Gus.  
**Confidence**: High
