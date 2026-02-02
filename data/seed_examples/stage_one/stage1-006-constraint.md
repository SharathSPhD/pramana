---
id: stage1-006
problem_type: constraint_satisfaction
difficulty: easy
variables: 3
ground_truth: "Quinn has the art book, Ravi has the history book, Sara has the math book"
metadata:
  created_date: 2026-02-02
  author: stage1
  validated: false
  z3_verifiable: false
  stage: 1
---

# Problem

Three students (Quinn, Ravi, Sara) each borrow one book: History, Math, or Art.

**Constraints**:
1. Sara has the Math book.
2. Quinn does not have the History book.
3. Ravi does not have the Math book.

**Question**: Which book does each student have?

---

## Samshaya (Doubt Analysis)

**Doubt Type**: Samana Dharma Upapatti (Multiple possibilities share similar properties)

**Justification**: The book assignments are uncertain until the exclusions force a unique allocation.

---

## Pramana (Sources of Knowledge)

### Pratyaksha (Direct Perception)
- Sara has the Math book.
- Quinn does not have the History book.
- Ravi does not have the Math book.
- Each student has exactly one book.

### Anumana (Inference)
- Since Sara has Math, Quinn and Ravi cannot have Math.
- Quinn cannot have History, so Quinn must have Art.
- Ravi then must have History.

### Upamana (Comparison)
- This mirrors a standard unique assignment with exclusions.

### Shabda (Testimony)
- If one option is fixed, remaining options distribute to remaining people.

---

## Pancha Avayava (5-Member Syllogism)

### Syllogism 1: Determine Quinn's Book
**Pratijna (Thesis)**: Quinn has the Art book.  
**Hetu (Reason)**: Math is assigned to Sara and Quinn cannot take History.  
**Udaharana (Universal + Example)**: Wherever only one option remains after exclusions, the remaining option must be chosen.  
**Upanaya (Application)**: Quinn cannot take Math or History, so Art remains.  
**Nigamana (Conclusion)**: Therefore, Quinn has the Art book.

---

## Tarka (Counterfactual Reasoning)

**Hypothesis**: Suppose Quinn does not have the Art book.  
**Consequence**: Quinn would need Math or History, but Math is assigned and History is forbidden.  
**Analysis**: This is a contradiction with the constraints.  
**Resolution**: Therefore, Quinn must have the Art book.

---

## Hetvabhasa (Fallacy Check)

Check for Savyabhichara: No  
Check for Viruddha: No  
Check for Asiddha: No  
Check for Satpratipaksha: No  
Check for Badhita: No

---

## Nirnaya (Ascertainment)

**Final Answer**: Quinn has the Art book, Ravi has the History book, Sara has the Math book.  
**Justification**: Sara is fixed to Math, Quinn cannot take History, leaving Art for Quinn and History for Ravi.  
**Confidence**: High
