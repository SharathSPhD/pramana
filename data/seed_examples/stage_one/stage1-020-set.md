---
id: stage1-020
problem_type: set_membership
difficulty: easy
variables: 3
ground_truth: "Uma is on Red, Vic is on Blue, Wren is on Green"
metadata:
  created_date: 2026-02-02
  author: stage1
  validated: false
  z3_verifiable: false
  stage: 1
---

# Problem

Three players (Uma, Vic, Wren) are assigned to teams: Red, Blue, or Green.

**Constraints**:
1. Vic is on Blue.
2. Uma is not on Green.
3. Wren is not on Blue.

**Question**: Which team does each player belong to?

---

## Samshaya (Doubt Analysis)

**Doubt Type**: Samana Dharma Upapatti (Multiple possibilities share similar properties)

**Justification**: The team assignment is uncertain until exclusions fix the arrangement.

---

## Pramana (Sources of Knowledge)

### Pratyaksha (Direct Perception)
- Vic is on Blue.
- Uma is not on Green.
- Wren is not on Blue.
- Each player is on exactly one team.

### Anumana (Inference)
- With Vic on Blue, Uma and Wren cannot be on Blue.
- Uma cannot be on Green, so Uma must be on Red.
- Wren then must be on Green.

### Upamana (Comparison)
- This is a unique assignment in a three-team partition.

### Shabda (Testimony)
- If one item is fixed, remaining items distribute to remaining people.

---

## Pancha Avayava (5-Member Syllogism)

### Syllogism 1: Determine Uma's Team
**Pratijna (Thesis)**: Uma is on Red.  
**Hetu (Reason)**: Blue is taken by Vic and Uma cannot be on Green.  
**Udaharana (Universal + Example)**: Wherever only one option remains after exclusions, the remaining option must be chosen.  
**Upanaya (Application)**: Uma cannot be on Blue or Green, so Red remains.  
**Nigamana (Conclusion)**: Therefore, Uma is on Red.

---

## Tarka (Counterfactual Reasoning)

**Hypothesis**: Suppose Uma is not on Red.  
**Consequence**: She would need Blue or Green, but Blue is taken and Green is forbidden.  
**Analysis**: This contradicts the constraints.  
**Resolution**: Therefore, Uma must be on Red.

---

## Hetvabhasa (Fallacy Check)

Check for Savyabhichara: No  
Check for Viruddha: No  
Check for Asiddha: No  
Check for Satpratipaksha: No  
Check for Badhita: No

---

## Nirnaya (Ascertainment)

**Final Answer**: Uma is on Red, Vic is on Blue, Wren is on Green.  
**Justification**: Vic is fixed to Blue, Uma cannot be Green, leaving Red for Uma and Green for Wren.  
**Confidence**: High
