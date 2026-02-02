---
id: stage1-005
problem_type: constraint_satisfaction
difficulty: easy
variables: 3
ground_truth: "Nora has the juice, Omar has the coffee, Pia has the tea"
metadata:
  created_date: 2026-02-02
  author: stage1
  validated: false
  z3_verifiable: false
  stage: 1
---

# Problem

Three coworkers (Nora, Omar, Pia) each choose one drink: tea, coffee, or juice.

**Constraints**:
1. Omar has the coffee.
2. Nora does not have the tea.
3. Pia does not have the coffee.

**Question**: Who has which drink?

---

## Samshaya (Doubt Analysis)

**Doubt Type**: Samana Dharma Upapatti (Multiple possibilities share similar properties)

**Justification**: The drink assignments are not obvious until we enforce the constraints.

---

## Pramana (Sources of Knowledge)

### Pratyaksha (Direct Perception)
- Omar has the coffee.
- Nora does not have the tea.
- Pia does not have the coffee.
- Each person has exactly one drink.

### Anumana (Inference)
- Since Omar has coffee, Nora and Pia cannot have coffee.
- Nora cannot have tea, so Nora must have juice.
- Pia then must have tea.

### Upamana (Comparison)
- This is a unique assignment problem with exclusions.

### Shabda (Testimony)
- If an item is fixed to one person, it is excluded for others.

---

## Pancha Avayava (5-Member Syllogism)

### Syllogism 1: Determine Nora's Drink
**Pratijna (Thesis)**: Nora has the juice.  
**Hetu (Reason)**: Coffee is assigned to Omar and Nora cannot have tea.  
**Udaharana (Universal + Example)**: Wherever only one option remains after exclusions, the remaining option must be chosen.  
**Upanaya (Application)**: Nora cannot take coffee or tea, so juice remains.  
**Nigamana (Conclusion)**: Therefore, Nora has the juice.

---

## Tarka (Counterfactual Reasoning)

**Hypothesis**: Suppose Nora does not have the juice.  
**Consequence**: She would need tea or coffee, but tea is forbidden and coffee is assigned.  
**Analysis**: This contradicts the constraints.  
**Resolution**: Therefore, Nora must have the juice.

---

## Hetvabhasa (Fallacy Check)

Check for Savyabhichara: No  
Check for Viruddha: No  
Check for Asiddha: No  
Check for Satpratipaksha: No  
Check for Badhita: No

---

## Nirnaya (Ascertainment)

**Final Answer**: Nora has the juice, Omar has the coffee, Pia has the tea.  
**Justification**: Omar is fixed to coffee, Nora cannot take tea, so Nora takes juice and Pia takes tea.  
**Confidence**: High
