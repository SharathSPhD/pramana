---
id: stage1-003
problem_type: constraint_satisfaction
difficulty: easy
variables: 3
ground_truth: "Hana sits in seat 1, Ivan sits in seat 2, Jae sits in seat 3"
metadata:
  created_date: 2026-02-02
  author: stage1
  validated: false
  z3_verifiable: false
  stage: 1
---

# Problem

Three people (Hana, Ivan, Jae) sit in seats 1, 2, and 3.

**Constraints**:
1. Ivan sits in seat 2.
2. Hana does not sit in seat 2.
3. Jae does not sit in seat 1.

**Question**: Where does each person sit?

---

## Samshaya (Doubt Analysis)

**Doubt Type**: Samana Dharma Upapatti (Multiple possibilities share similar properties)

**Justification**: Multiple seating arrangements are possible until the constraints fix the positions.

---

## Pramana (Sources of Knowledge)

### Pratyaksha (Direct Perception)
- Ivan sits in seat 2.
- Hana does not sit in seat 2.
- Jae does not sit in seat 1.
- Each person occupies exactly one seat.

### Anumana (Inference)
- With Ivan in seat 2, seats 1 and 3 remain for Hana and Jae.
- Jae cannot sit in seat 1, so Jae must sit in seat 3.
- Hana then must sit in seat 1.

### Upamana (Comparison)
- This is a direct placement puzzle with exclusions.

### Shabda (Testimony)
- Unique assignments follow from fixed positions and exclusions.

---

## Pancha Avayava (5-Member Syllogism)

### Syllogism 1: Determine Jae's Seat
**Pratijna (Thesis)**: Jae sits in seat 3.  
**Hetu (Reason)**: Ivan occupies seat 2 and Jae cannot sit in seat 1.  
**Udaharana (Universal + Example)**: Wherever a person is excluded from all but one seat, the remaining seat must be assigned to that person.  
**Upanaya (Application)**: Jae cannot sit in seat 1 and seat 2 is occupied, so seat 3 remains.  
**Nigamana (Conclusion)**: Therefore, Jae sits in seat 3.

---

## Tarka (Counterfactual Reasoning)

**Hypothesis**: Suppose Jae does not sit in seat 3.  
**Consequence**: Jae would need seat 1 or 2, but seat 2 is Ivan's and seat 1 is forbidden.  
**Analysis**: This is a contradiction with the constraints.  
**Resolution**: Therefore, Jae must sit in seat 3.

---

## Hetvabhasa (Fallacy Check)

Check for Savyabhichara: No  
Check for Viruddha: No  
Check for Asiddha: No  
Check for Satpratipaksha: No  
Check for Badhita: No

---

## Nirnaya (Ascertainment)

**Final Answer**: Hana sits in seat 1, Ivan sits in seat 2, Jae sits in seat 3.  
**Justification**: Ivan is fixed to seat 2 and Jae is excluded from seat 1, leaving seat 3 for Jae and seat 1 for Hana.  
**Confidence**: High
