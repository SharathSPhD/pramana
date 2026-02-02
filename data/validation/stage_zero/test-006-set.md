---
id: test-006
problem_type: set_membership
difficulty: easy
variables: 3
ground_truth: "Maya is in Math, Nikhil is in Science, Priya is in Art"
metadata:
  created_date: 2026-01-31
  author: validation
  validated: false
  z3_verifiable: true
  stage: 0
---

# Problem

Three students (Maya, Nikhil, Priya) each belong to exactly one club: Math, Science, or Art.

**Constraints**:
1. Maya is in the Math club.
2. Nikhil is not in the Art club.
3. Priya is not in the Science club.

**Question**: Which club does each student belong to?

---

## Samshaya (Doubt Analysis)

**Doubt Type**: Samana Dharma Upapatti (Multiple possibilities share similar properties)

**Justification**: Without applying the constraints, multiple club assignments seem possible. The doubt is resolved by elimination.

---

## Pramana (Sources of Knowledge)

### Pratyaksha (Direct Perception)
- Maya is in the Math club.
- Nikhil is not in the Art club.
- Priya is not in the Science club.
- Each student belongs to exactly one club.

### Anumana (Inference)
- If Maya is in Math, Science and Art remain for Nikhil and Priya.
- Nikhil cannot take Art, so Nikhil must take Science.
- The remaining club (Art) belongs to Priya.

### Upamana (Comparison)
- This is a standard one-to-one assignment with eliminations.

### Shabda (Testimony)
- When two options are eliminated, the remaining option is forced.

---

## Pancha Avayava (5-Member Syllogism)

### Syllogism 1: Fix Maya's Club
**Pratijna (Thesis)**: Maya is in Math.  
**Hetu (Reason)**: It is directly stated in constraint 1.  
**Udaharana (Universal + Example)**: Wherever a club membership is directly stated, it is accepted as true.  
**Upanaya (Application)**: The problem states Maya is in Math.  
**Nigamana (Conclusion)**: Therefore, Maya is in Math.

### Syllogism 2: Determine Nikhil's Club
**Pratijna (Thesis)**: Nikhil is in Science.  
**Hetu (Reason)**: Math is taken by Maya, and Nikhil cannot join Art.  
**Udaharana (Universal + Example)**: Wherever all but one option are eliminated, the remaining option must hold.  
**Upanaya (Application)**: Nikhil cannot be in Art and Math is occupied, leaving Science.  
**Nigamana (Conclusion)**: Therefore, Nikhil is in Science.

### Syllogism 3: Determine Priya's Club
**Pratijna (Thesis)**: Priya is in Art.  
**Hetu (Reason)**: Math and Science are assigned to Maya and Nikhil.  
**Udaharana (Universal + Example)**: Wherever two assignments are fixed, the remaining option goes to the remaining person.  
**Upanaya (Application)**: The only remaining club is Art.  
**Nigamana (Conclusion)**: Therefore, Priya is in Art.

---

## Tarka (Counterfactual Reasoning)

**Hypothesis**: Suppose Nikhil is not in Science.  
**Consequence**: Then Nikhil would have to be in Art or Math.  
**Analysis**: Math is already occupied by Maya, and Art is forbidden for Nikhil. This leaves no valid club for Nikhil.  
**Resolution**: Therefore, Nikhil must be in Science.

---

## Hetvabhasa (Fallacy Check)

Check for Savyabhichara: No  
Check for Viruddha: No  
Check for Asiddha: No  
Check for Satpratipaksha: No  
Check for Badhita: No  

---

## Nirnaya (Ascertainment)

**Final Answer**: Maya is in Math, Nikhil is in Science, and Priya is in Art.  
**Justification**: Maya is fixed in Math. Nikhil cannot be in Art, so he is in Science. Priya receives the remaining club.  
**Confidence**: High
