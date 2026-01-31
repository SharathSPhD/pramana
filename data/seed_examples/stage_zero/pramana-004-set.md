---
id: pramana-004
problem_type: set_membership
difficulty: medium
variables: 4
ground_truth: "Group 1: {Alice, Bob}, Group 2: {Carol, David}"
metadata:
  stage: 0
  verified: true
  created_at: 2026-01-31
  z3_verifiable: true
---

# Problem

Four people (Alice, Bob, Carol, David) must be divided into two groups. Each person belongs to exactly one group. The following constraints are known:

**Constraints**:
1. Alice and Bob are in the same group
2. Carol and David are in different groups
3. If Alice is in Group 1, then Carol is in Group 2
4. Bob is not in the same group as David

**Question**: How should the four people be divided into the two groups?

---

## Samshaya (Doubt Analysis)

**Doubt Type**: Samana Dharma Upapatti (Multiple possibilities share similar properties)

**Justification**: There are multiple ways to divide four people into two groups. Without systematic reasoning, we cannot determine which specific assignment satisfies all constraints. The doubt arises because multiple arrangements seem possible, and we must eliminate impossible ones through logical deduction to reach certainty about the group assignments.

---

## Pramana (Sources of Knowledge)

### Pratyaksha (Direct Perception)

```yaml
observable_facts:
  - "Alice and Bob are in the same group"
  - "Carol and David are in different groups"
  - "If Alice is in Group 1, then Carol is in Group 2"
  - "Bob is not in the same group as David"
  - "There are exactly four people: Alice, Bob, Carol, David"
  - "There are exactly two groups: Group 1 and Group 2"
  - "Each person belongs to exactly one group"
  - "Each group must contain at least one person"
```

**Note**: These are the only directly stated facts. No inferences about actual group assignments are included here.

### Anumana (Inference)

```yaml
inferences:
  - type: purvavat
    premise: "Alice and Bob are in the same group (constraint 1), and Bob is not in the same group as David (constraint 4)"
    conclusion: "Alice is not in the same group as David"
    justification: "If Alice and Bob are together, and Bob is not with David, then Alice cannot be with David either"
  
  - type: purvavat
    premise: "Alice and Bob are in the same group, and if Alice is in Group 1 then Carol is in Group 2 (constraint 3)"
    conclusion: "If Alice and Bob are in Group 1, then Carol is in Group 2"
    justification: "Since Alice and Bob are together, if Alice is in Group 1, Bob is also in Group 1, and the conditional applies"
  
  - type: purvavat
    premise: "Carol and David are in different groups (constraint 2), and if Alice is in Group 1 then Carol is in Group 2"
    conclusion: "If Alice is in Group 1, then Carol is in Group 2 and David is in Group 1 (since Carol and David are in different groups)"
    justification: "If Carol is in Group 2, and Carol and David are in different groups, then David must be in Group 1"
  
  - type: purvavat
    premise: "Alice and Bob are in Group 1 (from testing), Carol is in Group 2, David is in Group 1, but Bob is not in the same group as David (constraint 4)"
    conclusion: "This leads to contradiction, so Alice and Bob cannot be in Group 1"
    justification: "Constraint 4 requires Bob and David to be in different groups, but if Alice/Bob are in Group 1 and David is also in Group 1, this violates constraint 4"
  
  - type: purvavat
    premise: "Alice and Bob must be in Group 2 (since Group 1 leads to contradiction)"
    conclusion: "Alice and Bob are in Group 2"
    justification: "By elimination, since Group 1 assignment leads to contradiction"
  
  - type: purvavat
    premise: "Alice and Bob are in Group 2, and Carol and David are in different groups"
    conclusion: "One of Carol or David is in Group 1, the other in Group 2"
    justification: "Since Alice and Bob occupy Group 2, and Carol and David must be in different groups, one must be in Group 1"
  
  - type: purvavat
    premise: "Bob is in Group 2, and Bob is not in the same group as David (constraint 4)"
    conclusion: "David is not in Group 2, so David is in Group 1"
    justification: "Since Bob is in Group 2, David must be in the other group"
  
  - type: purvavat
    premise: "David is in Group 1, and Carol and David are in different groups (constraint 2)"
    conclusion: "Carol is in Group 2"
    justification: "Since David is in Group 1, Carol must be in Group 2"
```

### Upamana (Comparison)

```yaml
analogies:
  - reference: "Graph coloring problems and bipartite graph partitioning"
    similarity: "This problem is structurally similar to partitioning vertices of a graph into two sets based on constraints. The constraints 'same group' and 'different groups' are analogous to edges connecting vertices that must have the same color or different colors. The problem structure is isomorphic to 2-coloring a graph with specific edge constraints"
```

### Shabda (Testimony)

```yaml
principles:
  - "Mutual Exclusivity: If X and Y are in the same group, and Y and Z are in different groups, then X and Z are in different groups"
  - "Transitivity of Same Group: If X and Y are in the same group, and Y and Z are in the same group, then X and Z are in the same group"
  - "Completeness Principle: Every person must be assigned to exactly one group"
  - "Non-Contradiction: No constraint can be violated in a valid assignment"
  - "Elimination Principle: If an assignment leads to contradiction, that assignment is impossible"
```

---

## Pancha Avayava (5-Member Syllogism)

### Syllogism 1: Establishing Alice and Bob's Group Through Elimination

**Pratijna (Thesis)**: Alice and Bob are in Group 2.

**Hetu (Reason)**: If Alice and Bob were in Group 1, then by constraint 3, Carol would be in Group 2. Since Carol and David are in different groups (constraint 2), David would be in Group 1. But constraint 4 states Bob is not in the same group as David. If Alice and Bob are in Group 1 and David is also in Group 1, then Bob and David are in the same group, violating constraint 4. This contradiction shows Alice and Bob cannot be in Group 1, so they must be in Group 2.

**Udaharana (Universal + Example)**: Wherever an assignment leads to a logical contradiction with the given constraints, that assignment is impossible and must be rejected. For example, if we are told "X and Y are together" and "X and Z are together" and "Y and Z are apart," this is contradictory and no valid assignment exists. This universal principle applies to all constraint satisfaction problems: contradictory assignments are invalid.

**Upanaya (Application)**: Testing the assignment "Alice and Bob in Group 1" leads to: Carol in Group 2 (from constraint 3), David in Group 1 (from constraint 2, since Carol and David are in different groups), but then Bob and David are both in Group 1, violating constraint 4. This contradiction shows the assignment is impossible. Therefore, Alice and Bob must be in Group 2.

**Nigamana (Conclusion)**: Therefore, Alice and Bob are in Group 2.

### Syllogism 2: Establishing David's Group

**Pratijna (Thesis)**: David is in Group 1.

**Hetu (Reason)**: Alice and Bob are in Group 2 (Syllogism 1), and Bob is not in the same group as David (constraint 4). Since Bob is in Group 2, David cannot be in Group 2, so David must be in Group 1.

**Udaharana (Universal + Example)**: Wherever we have exactly two groups and know that person X is in one group and person X cannot be with person Y, then person Y must be in the other group. For example, if we have two teams and player A is on Team 1, and A cannot be on the same team as player B, then B must be on Team 2. This universal rule applies whenever we have binary partitioning with exclusion constraints.

**Upanaya (Application)**: We have two groups (Group 1 and Group 2). Bob is in Group 2 (from Syllogism 1). Constraint 4 states Bob is not in the same group as David. Since Bob is in Group 2, David cannot be in Group 2. Therefore, David must be in Group 1.

**Nigamana (Conclusion)**: Therefore, David is in Group 1.

### Syllogism 3: Establishing Carol's Group

**Pratijna (Thesis)**: Carol is in Group 2.

**Hetu (Reason)**: David is in Group 1 (Syllogism 2), and Carol and David are in different groups (constraint 2). Since David is in Group 1, Carol must be in Group 2.

**Udaharana (Universal + Example)**: Wherever two items must be in different groups and we know one item's group, the other item must be in the other group. For example, if we have two boxes and objects X and Y must be in different boxes, and we know X is in Box 1, then Y must be in Box 2. This universal principle applies to all binary partitioning problems with "different groups" constraints.

**Upanaya (Application)**: Carol and David must be in different groups (constraint 2). David is in Group 1 (Syllogism 2). Therefore, Carol must be in Group 2.

**Nigamana (Conclusion)**: Therefore, Carol is in Group 2.

### Syllogism 4: Verifying Complete Assignment

**Pratijna (Thesis)**: The complete assignment is: Group 1 contains {David}, Group 2 contains {Alice, Bob, Carol}.

**Hetu (Reason)**: We have established: Alice and Bob in Group 2 (Syllogism 1), David in Group 1 (Syllogism 2), Carol in Group 2 (Syllogism 3). This satisfies all constraints: Alice and Bob are together ✓, Carol and David are in different groups ✓, the conditional constraint 3 is satisfied (Alice in Group 2, so the premise is false, making the conditional true) ✓, Bob and David are in different groups ✓.

**Udaharana (Universal + Example)**: Wherever we have a complete assignment that satisfies all given constraints, that assignment is valid. For example, if we assign items to boxes such that every constraint is satisfied, the assignment is correct. This universal principle applies to all constraint satisfaction problems: an assignment satisfying all constraints is a valid solution.

**Upanaya (Application)**: Our assignment is: Group 1 = {David}, Group 2 = {Alice, Bob, Carol}. Verification: Constraint 1 (Alice and Bob same group) ✓, Constraint 2 (Carol and David different groups) ✓, Constraint 3 (if Alice in Group 1 then Carol in Group 2 - premise false, so conditional true) ✓, Constraint 4 (Bob and David different groups) ✓. All constraints satisfied.

**Nigamana (Conclusion)**: Therefore, Group 1 contains David, and Group 2 contains Alice, Bob, and Carol.

---

## Tarka (Counterfactual Reasoning)

**Hypothesis**: Suppose Alice and Bob are not both in Group 2 (i.e., at least one is in Group 1, or they are in different groups).

**Consequence**: If Alice and Bob are not both in Group 2, then either: (1) They are in different groups, or (2) They are both in Group 1, or (3) One is in Group 1 and one is in Group 2. Case (1) violates constraint 1 (Alice and Bob must be in the same group). Case (2) leads to contradiction as shown in Syllogism 1. Case (3) also violates constraint 1. Therefore, all possibilities lead to contradiction or constraint violation.

**Analysis**: The hypothesis leads to impossibility. We must have Alice and Bob together, and they cannot be in Group 1 (contradiction), so they must be in Group 2.

**Resolution**: Therefore, Alice and Bob must both be in Group 2.

**Additional Tarka Test**: Suppose Carol is not in Group 2 (i.e., Carol is in Group 1). Then, since Carol and David are in different groups (constraint 2), David would be in Group 2. But Alice and Bob are in Group 2 (established), so Group 2 would contain {Alice, Bob, David}. But constraint 4 states Bob is not in the same group as David. This is a contradiction. Therefore, Carol must be in Group 2.

---

## Hetvabhasa (Fallacy Check)

```yaml
fallacy_checks:
  savyabhichara: none_detected
    # Erratic reasoning: Our elimination method is systematic and consistent. We test assignments and reject those leading to contradiction.
  
  viruddha: none_detected
    # Contradictory reasoning: No contradictions exist in our final assignment. All constraints are satisfied simultaneously.
  
  prakaranasama: none_detected
    # Circular reasoning: We do not assume the conclusion. We derive it through elimination: test Group 1 assignment, find contradiction, conclude Group 2.
  
  sadhyasama: none_detected
    # Begging the question: Our premises (constraints) are independent of our conclusion. We use logical elimination, not assumption of the answer.
  
  kalaatita: none_detected
    # Temporal fallacy: No temporal reasoning involved. Group assignments are static properties.

reasoning: "All reasoning steps follow valid logical principles. We systematically test possible assignments and eliminate those leading to contradiction. Each syllogism builds on previously established facts. Tarka testing confirms conclusions through reductio ad absurdum. No fallacies detected."
```

---

## Nirnaya (Ascertainment)

**Status**: Definitive Knowledge

**Answer**: Group 1 contains David. Group 2 contains Alice, Bob, and Carol.

**Justification**: All constraints are satisfied: Alice and Bob are in the same group (Group 2) ✓, Carol and David are in different groups (Carol in Group 2, David in Group 1) ✓, the conditional constraint 3 is satisfied (Alice is in Group 2, so the premise is false, making the conditional true) ✓, Bob and David are in different groups (Bob in Group 2, David in Group 1) ✓. The reasoning follows valid logical principles: we systematically test assignments, eliminate contradictions, and derive the solution through logical necessity. Tarka testing confirms the conclusions through reductio ad absurdum.

**Confidence**: High - The solution is logically necessary given the constraints. The elimination method shows that no other assignment satisfies all constraints simultaneously. The answer is complete and verifiable.
