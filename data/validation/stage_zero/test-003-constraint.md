---
id: test-003
problem_type: constraint_satisfaction
difficulty: easy
variables: 3
ground_truth: "Liam has the green card, Mia has the red card, Noah has the blue card"
metadata:
  created_date: 2026-01-31
  author: validation
  validated: false
  z3_verifiable: true
  stage: 0
---

# Problem

Three students (Liam, Mia, Noah) each receive one colored card: red, blue, or green.

**Constraints**:
1. Noah has the blue card.
2. Liam does not have the red card.
3. Mia does not have the green card.

**Question**: Which color does each student have?

---

## Samshaya (Doubt Analysis)

**Doubt Type**: Samana Dharma Upapatti (Multiple possibilities share similar properties)

**Justification**: With three students and three colors, several assignments are possible until the constraints eliminate options.

---

## Pramana (Sources of Knowledge)

### Pratyaksha (Direct Perception)
- Noah has the blue card.
- Liam does not have the red card.
- Mia does not have the green card.
- Each student has exactly one card.

### Anumana (Inference)
- If Noah has blue, red and green remain for Liam and Mia.
- Liam cannot take red, so Liam must take green.
- The remaining color (red) goes to Mia.

### Upamana (Comparison)
- This follows the same elimination pattern as assigning unique labels to a fixed set.

### Shabda (Testimony)
- Unique assignment plus elimination yields the remaining option.

---

## Pancha Avayava (5-Member Syllogism)

### Syllogism 1: Fix Noah's Color
**Pratijna (Thesis)**: Noah has the blue card.  
**Hetu (Reason)**: It is directly stated in constraint 1.  
**Udaharana (Universal + Example)**: Wherever a direct assignment is given, it is accepted as true.  
**Upanaya (Application)**: The problem states Noah has blue.  
**Nigamana (Conclusion)**: Therefore, Noah has the blue card.

### Syllogism 2: Determine Liam's Color
**Pratijna (Thesis)**: Liam has the green card.  
**Hetu (Reason)**: Red is excluded for Liam, and blue is taken by Noah.  
**Udaharana (Universal + Example)**: Wherever all but one option are eliminated, the remaining option must hold.  
**Upanaya (Application)**: Liam cannot take red and blue is assigned, leaving green.  
**Nigamana (Conclusion)**: Therefore, Liam has the green card.

### Syllogism 3: Determine Mia's Color
**Pratijna (Thesis)**: Mia has the red card.  
**Hetu (Reason)**: Blue is Noah's and green is Liam's, leaving red.  
**Udaharana (Universal + Example)**: Wherever two assignments are fixed, the remaining item goes to the remaining person.  
**Upanaya (Application)**: The only unassigned color is red, and Mia is the remaining student.  
**Nigamana (Conclusion)**: Therefore, Mia has the red card.

---

## Tarka (Counterfactual Reasoning)

**Hypothesis**: Suppose Liam does not have the green card.  
**Consequence**: Liam would have to take red or blue.  
**Analysis**: Blue is already assigned to Noah, and red is forbidden for Liam. This leaves no valid card for Liam, which is impossible.  
**Resolution**: Therefore, Liam must have the green card.

---

## Hetvabhasa (Fallacy Check)

Check for Savyabhichara: No  
Check for Viruddha: No  
Check for Asiddha: No  
Check for Satpratipaksha: No  
Check for Badhita: No  

---

## Nirnaya (Ascertainment)

**Final Answer**: Liam has the green card, Mia has the red card, and Noah has the blue card.  
**Justification**: Noah's assignment is direct, Liam's is forced by elimination, and Mia receives the remaining color.  
**Confidence**: High
