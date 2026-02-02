---
id: test-007
problem_type: set_membership
difficulty: easy
variables: 3
ground_truth: "Shelf A has the Math book, Shelf B has the History book, Shelf C has the Physics book"
metadata:
  created_date: 2026-01-31
  author: validation
  validated: false
  z3_verifiable: true
  stage: 0
---

# Problem

Three shelves (A, B, C) each hold one book: History, Math, or Physics.

**Constraints**:
1. Shelf B has the History book.
2. Shelf A does not have the Physics book.
3. Shelf C does not have the Math book.

**Question**: Which book is on each shelf?

---

## Samshaya (Doubt Analysis)

**Doubt Type**: Samana Dharma Upapatti (Multiple possibilities share similar properties)

**Justification**: Multiple book placements are possible until the constraints eliminate options.

---

## Pramana (Sources of Knowledge)

### Pratyaksha (Direct Perception)
- Shelf B has the History book.
- Shelf A does not have the Physics book.
- Shelf C does not have the Math book.
- Each shelf has exactly one book.

### Anumana (Inference)
- With History fixed on B, Math and Physics remain for A and C.
- A cannot take Physics, so A must take Math.
- The remaining book (Physics) goes to C.

### Upamana (Comparison)
- This is a standard elimination puzzle with unique assignments.

### Shabda (Testimony)
- If all but one option are eliminated, the remaining option must be chosen.

---

## Pancha Avayava (5-Member Syllogism)

### Syllogism 1: Fix Shelf B
**Pratijna (Thesis)**: Shelf B has the History book.  
**Hetu (Reason)**: It is directly stated in constraint 1.  
**Udaharana (Universal + Example)**: Wherever a direct assignment is given, it is accepted as true.  
**Upanaya (Application)**: The problem states Shelf B has History.  
**Nigamana (Conclusion)**: Therefore, Shelf B has the History book.

### Syllogism 2: Determine Shelf A
**Pratijna (Thesis)**: Shelf A has the Math book.  
**Hetu (Reason)**: History is taken by B and A cannot take Physics.  
**Udaharana (Universal + Example)**: Wherever all but one option are eliminated, the remaining option must hold.  
**Upanaya (Application)**: A cannot take Physics and History is taken, leaving Math.  
**Nigamana (Conclusion)**: Therefore, Shelf A has the Math book.

### Syllogism 3: Determine Shelf C
**Pratijna (Thesis)**: Shelf C has the Physics book.  
**Hetu (Reason)**: History is on B and Math is on A, leaving Physics.  
**Udaharana (Universal + Example)**: Wherever two assignments are fixed, the remaining item goes to the remaining location.  
**Upanaya (Application)**: The only remaining book is Physics, and Shelf C is the remaining shelf.  
**Nigamana (Conclusion)**: Therefore, Shelf C has the Physics book.

---

## Tarka (Counterfactual Reasoning)

**Hypothesis**: Suppose Shelf A does not have the Math book.  
**Consequence**: Then A would have to take Physics.  
**Analysis**: This contradicts constraint 2, which forbids Physics on A.  
**Resolution**: Therefore, Shelf A must have the Math book.

---

## Hetvabhasa (Fallacy Check)

Check for Savyabhichara: No  
Check for Viruddha: No  
Check for Asiddha: No  
Check for Satpratipaksha: No  
Check for Badhita: No  

---

## Nirnaya (Ascertainment)

**Final Answer**: Shelf A has the Math book, Shelf B has the History book, and Shelf C has the Physics book.  
**Justification**: The direct assignment fixes History on B. A cannot take Physics, leaving Math. The remaining book, Physics, goes to C.  
**Confidence**: High
