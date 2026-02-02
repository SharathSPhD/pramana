---
id: stage1-026
problem_type: multi_step_deduction
difficulty: easy
variables: 4
ground_truth: "Payment received, order processed, shipment scheduled, tracking sent"
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
1. If payment is received, then the order is processed.
2. If the order is processed, then shipment is scheduled.
3. If shipment is scheduled, then tracking is sent.
4. Payment is received.

**Question**: What can we conclude about the order, shipment, and tracking?

---

## Samshaya (Doubt Analysis)

**Doubt Type**: Vipratipatti (Conflicting possibilities to resolve)

**Justification**: We must propagate the consequences of payment through the chain.

---

## Pramana (Sources of Knowledge)

### Pratyaksha (Direct Perception)
- Payment received → order processed.
- Order processed → shipment scheduled.
- Shipment scheduled → tracking sent.
- Payment is received.

### Anumana (Inference)
- From payment, infer order processed.
- From order processed, infer shipment scheduled.
- From shipment scheduled, infer tracking sent.

### Upamana (Comparison)
- This is a standard implication chain.

### Shabda (Testimony)
- If A → B and A is true, then B is true.

---

## Pancha Avayava (5-Member Syllogism)

### Syllogism 1: Derive Tracking Sent
**Pratijna (Thesis)**: Tracking is sent.  
**Hetu (Reason)**: Payment triggers order, order triggers shipment, shipment triggers tracking.  
**Udaharana (Universal + Example)**: Wherever implications chain and the first antecedent holds, all consequents follow.  
**Upanaya (Application)**: Payment is received, so the chain implies tracking is sent.  
**Nigamana (Conclusion)**: Therefore, tracking is sent.

---

## Tarka (Counterfactual Reasoning)

**Hypothesis**: Suppose tracking is not sent.  
**Consequence**: Then shipment was not scheduled, which would contradict payment being received through the chain.  
**Analysis**: This contradicts the given implications.  
**Resolution**: Therefore, tracking must be sent.

---

## Hetvabhasa (Fallacy Check)

Check for Savyabhichara: No  
Check for Viruddha: No  
Check for Asiddha: No  
Check for Satpratipaksha: No  
Check for Badhita: No

---

## Nirnaya (Ascertainment)

**Final Answer**: Payment received, order processed, shipment scheduled, tracking sent.  
**Justification**: The implications chain forward from payment.  
**Confidence**: High
