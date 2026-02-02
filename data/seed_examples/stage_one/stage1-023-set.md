---
id: stage1-023
problem_type: set_membership
difficulty: easy
variables: 3
ground_truth: "Yara takes Math, Zane takes Science, Alan takes Art"
metadata:
  created_date: 2026-02-02
  author: stage1
  validated: false
  z3_verifiable: false
  stage: 1
---

# Problem

Three students (Yara, Zane, Alan) each choose one subject: Math, Science, or Art.

**Constraints**:
1. Zane takes Science.
2. Yara does not take Art.
3. Alan does not take Science.

**Question**: Which subject does each student take?

---

## Samshaya (Doubt Analysis)

**Doubt Type**: Samana Dharma Upapatti (Multiple possibilities share similar properties)

**Justification**: The subject assignments are uncertain until exclusions fix the choices.

---

## Pramana (Sources of Knowledge)

### Pratyaksha (Direct Perception)
- Zane takes Science.
- Yara does not take Art.
- Alan does not take Science.
- Each student takes exactly one subject.

### Anumana (Inference)
- Since Zane takes Science, Yara and Alan cannot take Science.
- Yara cannot take Art, so Yara must take Math.
- Alan then must take Art.

### Upamana (Comparison)
- This is a direct one-to-one assignment.

### Shabda (Testimony)
- If one subject is fixed, remaining subjects distribute to remaining students.

---

## Pancha Avayava (5-Member Syllogism)

### Syllogism 1: Determine Yara's Subject
**Pratijna (Thesis)**: Yara takes Math.  
**Hetu (Reason)**: Science is assigned to Zane and Yara cannot take Art.  
**Udaharana (Universal + Example)**: Wherever only one option remains after exclusions, the remaining option must be chosen.  
**Upanaya (Application)**: Yara cannot take Science or Art, so Math remains.  
**Nigamana (Conclusion)**: Therefore, Yara takes Math.

---

## Tarka (Counterfactual Reasoning)

**Hypothesis**: Suppose Yara does not take Math.  
**Consequence**: She would need Art or Science, but Art is forbidden and Science is taken.  
**Analysis**: This contradicts the constraints.  
**Resolution**: Therefore, Yara must take Math.

---

## Hetvabhasa (Fallacy Check)

Check for Savyabhichara: No  
Check for Viruddha: No  
Check for Asiddha: No  
Check for Satpratipaksha: No  
Check for Badhita: No

---

## Nirnaya (Ascertainment)

**Final Answer**: Yara takes Math, Zane takes Science, Alan takes Art.  
**Justification**: Zane is fixed to Science, Yara cannot take Art, so Yara takes Math and Alan takes Art.  
**Confidence**: High
