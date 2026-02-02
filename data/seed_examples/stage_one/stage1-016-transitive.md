---
id: stage1-016
problem_type: transitive_reasoning
difficulty: easy
variables: 3
ground_truth: "Ranking: Lina > Max > Nia"
metadata:
  created_date: 2026-02-02
  author: stage1
  validated: false
  z3_verifiable: false
  stage: 1
---

# Problem

Three sprinters (Lina, Max, Nia) are compared by speed.

**Constraints**:
1. Lina is faster than Max.
2. Max is faster than Nia.
3. Lina is faster than Nia.

**Question**: What is the complete speed ranking from fastest to slowest?

---

## Samshaya (Doubt Analysis)

**Doubt Type**: Vipratipatti (Conflicting possibilities to resolve)

**Justification**: We must integrate the comparisons into a total order.

---

## Pramana (Sources of Knowledge)

### Pratyaksha (Direct Perception)
- Lina is faster than Max.
- Max is faster than Nia.
- Lina is faster than Nia.

### Anumana (Inference)
- The relations imply Lina > Max > Nia.

### Upamana (Comparison)
- This is a transitive ranking problem.

### Shabda (Testimony)
- If A > B and B > C, then A > C.

---

## Pancha Avayava (5-Member Syllogism)

### Syllogism 1: Establish Order
**Pratijna (Thesis)**: Lina > Max > Nia.  
**Hetu (Reason)**: Lina > Max and Max > Nia.  
**Udaharana (Universal + Example)**: Wherever a relation is transitive, chaining yields a total order.  
**Upanaya (Application)**: Apply transitivity to the given comparisons.  
**Nigamana (Conclusion)**: Therefore, Lina > Max > Nia.

---

## Tarka (Counterfactual Reasoning)

**Hypothesis**: Suppose Nia were faster than Max.  
**Consequence**: This contradicts the given fact that Max is faster than Nia.  
**Analysis**: The hypothesis conflicts with the constraints.  
**Resolution**: Therefore, Max remains faster than Nia.

---

## Hetvabhasa (Fallacy Check)

Check for Savyabhichara: No  
Check for Viruddha: No  
Check for Asiddha: No  
Check for Satpratipaksha: No  
Check for Badhita: No

---

## Nirnaya (Ascertainment)

**Final Answer**: Lina > Max > Nia.  
**Justification**: The given comparisons form a consistent chain.  
**Confidence**: High
