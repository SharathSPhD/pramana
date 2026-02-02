---
id: stage1-024
problem_type: set_membership
difficulty: easy
variables: 3
ground_truth: "Card 1 has the circle, Card 2 has the square, Card 3 has the triangle"
metadata:
  created_date: 2026-02-02
  author: stage1
  validated: false
  z3_verifiable: false
  stage: 1
---

# Problem

Three cards (1, 2, 3) each show one symbol: circle, square, or triangle.

**Constraints**:
1. Card 3 has the triangle.
2. Card 1 does not have the square.
3. Card 2 does not have the triangle.

**Question**: Which symbol is on each card?

---

## Samshaya (Doubt Analysis)

**Doubt Type**: Samana Dharma Upapatti (Multiple possibilities share similar properties)

**Justification**: The symbol placement is uncertain until the exclusions fix each card.

---

## Pramana (Sources of Knowledge)

### Pratyaksha (Direct Perception)
- Card 3 has the triangle.
- Card 1 does not have the square.
- Card 2 does not have the triangle.
- Each card has exactly one symbol.

### Anumana (Inference)
- Since Card 3 has triangle, Cards 1 and 2 cannot have triangle.
- Card 1 cannot have square, so Card 1 must have circle.
- Card 2 then must have square.

### Upamana (Comparison)
- This is a standard exclusive assignment with three symbols.

### Shabda (Testimony)
- If one symbol is fixed, the remaining symbols distribute to remaining cards.

---

## Pancha Avayava (5-Member Syllogism)

### Syllogism 1: Determine Card 1
**Pratijna (Thesis)**: Card 1 has the circle.  
**Hetu (Reason)**: Triangle is on Card 3 and Card 1 cannot have square.  
**Udaharana (Universal + Example)**: Wherever only one option remains after exclusions, the remaining option must hold.  
**Upanaya (Application)**: Card 1 cannot have triangle or square, so circle remains.  
**Nigamana (Conclusion)**: Therefore, Card 1 has the circle.

---

## Tarka (Counterfactual Reasoning)

**Hypothesis**: Suppose Card 1 does not have the circle.  
**Consequence**: Card 1 would need square or triangle, but square is forbidden and triangle is on Card 3.  
**Analysis**: This contradicts the constraints.  
**Resolution**: Therefore, Card 1 must have the circle.

---

## Hetvabhasa (Fallacy Check)

Check for Savyabhichara: No  
Check for Viruddha: No  
Check for Asiddha: No  
Check for Satpratipaksha: No  
Check for Badhita: No

---

## Nirnaya (Ascertainment)

**Final Answer**: Card 1 has the circle, Card 2 has the square, Card 3 has the triangle.  
**Justification**: Triangle is fixed to Card 3, Card 1 cannot take square, leaving circle for Card 1 and square for Card 2.  
**Confidence**: High
