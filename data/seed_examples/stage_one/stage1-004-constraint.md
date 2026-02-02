---
id: stage1-004
problem_type: constraint_satisfaction
difficulty: easy
variables: 3
ground_truth: "Kira has the hammer, Leo has the saw, Mira has the drill"
metadata:
  created_date: 2026-02-02
  author: stage1
  validated: false
  z3_verifiable: false
  stage: 1
---

# Problem

Three workers (Kira, Leo, Mira) each use one tool: a hammer, a saw, or a drill.

**Constraints**:
1. Mira has the drill.
2. Kira does not have the saw.
3. Leo does not have the drill.

**Question**: Who has which tool?

---

## Samshaya (Doubt Analysis)

**Doubt Type**: Samana Dharma Upapatti (Multiple possibilities share similar properties)

**Justification**: The tools must be assigned uniquely, but exclusions create initial uncertainty.

---

## Pramana (Sources of Knowledge)

### Pratyaksha (Direct Perception)
- Mira has the drill.
- Kira does not have the saw.
- Leo does not have the drill.
- Each person has exactly one tool.

### Anumana (Inference)
- Since Mira has the drill, Kira and Leo cannot have the drill.
- Leo cannot have the drill and Kira cannot have the saw, so Kira must have the hammer.
- Leo then has the saw.

### Upamana (Comparison)
- This follows standard exclusive assignment logic.

### Shabda (Testimony)
- If one item is assigned, the remaining items must go to the remaining people.

---

## Pancha Avayava (5-Member Syllogism)

### Syllogism 1: Determine Kira's Tool
**Pratijna (Thesis)**: Kira has the hammer.  
**Hetu (Reason)**: The drill is assigned to Mira and Kira cannot take the saw.  
**Udaharana (Universal + Example)**: Wherever only one option remains after exclusions, the remaining option must be selected.  
**Upanaya (Application)**: Kira cannot take the saw and the drill is taken, so the hammer remains.  
**Nigamana (Conclusion)**: Therefore, Kira has the hammer.

---

## Tarka (Counterfactual Reasoning)

**Hypothesis**: Suppose Kira does not have the hammer.  
**Consequence**: Kira would need the saw or drill, but both are excluded.  
**Analysis**: This contradicts the constraints.  
**Resolution**: Therefore, Kira must have the hammer.

---

## Hetvabhasa (Fallacy Check)

Check for Savyabhichara: No  
Check for Viruddha: No  
Check for Asiddha: No  
Check for Satpratipaksha: No  
Check for Badhita: No

---

## Nirnaya (Ascertainment)

**Final Answer**: Kira has the hammer, Leo has the saw, Mira has the drill.  
**Justification**: Mira is fixed to the drill, Kira cannot use the saw, so Kira takes the hammer and Leo takes the saw.  
**Confidence**: High
