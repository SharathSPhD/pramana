---
id: stage1-029
problem_type: multi_step_deduction
difficulty: easy
variables: 4
ground_truth: "Exam passed, certificate issued, job eligible, interview scheduled"
metadata:
  created_date: 2026-02-02
  author: stage1
  validated: false
  z3_verifiable: false
  stage: 1
---

# Problem

Consider the following statements:

**Given Facts**:
1. If an exam is passed, then a certificate is issued.
2. If a certificate is issued, then the candidate is job-eligible.
3. If the candidate is job-eligible, then an interview is scheduled.
4. The exam is passed.

**Question**: What can we conclude about the certificate, eligibility, and interview?

---

## Samshaya (Doubt Analysis)

**Doubt Type**: Vipratipatti (Conflicting possibilities to resolve)

**Justification**: We must propagate the consequences of passing the exam.

---

## Pramana (Sources of Knowledge)

### Pratyaksha (Direct Perception)
- Exam passed → certificate issued.
- Certificate issued → job eligible.
- Job eligible → interview scheduled.
- Exam passed.

### Anumana (Inference)
- From exam passed, infer certificate issued.
- From certificate issued, infer job eligible.
- From job eligible, infer interview scheduled.

### Upamana (Comparison)
- This is a direct implication chain.

### Shabda (Testimony)
- If A → B and A is true, then B is true.

---

## Pancha Avayava (5-Member Syllogism)

### Syllogism 1: Derive Interview Scheduled
**Pratijna (Thesis)**: An interview is scheduled.  
**Hetu (Reason)**: Exam passed leads to certificate, then eligibility, then interview.  
**Udaharana (Universal + Example)**: Wherever implications chain and the first antecedent holds, all consequents follow.  
**Upanaya (Application)**: Exam passed implies certificate, eligibility, and interview.  
**Nigamana (Conclusion)**: Therefore, an interview is scheduled.

---

## Tarka (Counterfactual Reasoning)

**Hypothesis**: Suppose the interview is not scheduled.  
**Consequence**: Then job eligibility would fail, contradicting the chain from exam passed.  
**Analysis**: This contradicts the given implications.  
**Resolution**: Therefore, the interview must be scheduled.

---

## Hetvabhasa (Fallacy Check)

Check for Savyabhichara: No  
Check for Viruddha: No  
Check for Asiddha: No  
Check for Satpratipaksha: No  
Check for Badhita: No

---

## Nirnaya (Ascertainment)

**Final Answer**: Exam passed, certificate issued, job eligible, interview scheduled.  
**Justification**: The implications chain forward from passing the exam.  
**Confidence**: High
