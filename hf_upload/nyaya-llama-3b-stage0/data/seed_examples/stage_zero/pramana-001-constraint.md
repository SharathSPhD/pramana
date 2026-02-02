---
id: pramana-001
problem_type: constraint_satisfaction
difficulty: medium
variables: 3
ground_truth: "Alice has the fish, Bob has the dog, Carol has the cat"
metadata:
  stage: 0
  verified: true
  created_at: 2026-01-31
  z3_verifiable: true
---

# Problem

Three people (Alice, Bob, Carol) each have one pet: a cat, a dog, or a fish. The following constraints are known:

**Constraints**:
1. Alice does not have the cat
2. Bob has the dog
3. Carol does not have the fish

**Question**: Who has which pet?

---

## Samshaya (Doubt Analysis)

**Doubt Type**: Samana Dharma Upapatti (Multiple possibilities share similar properties)

**Justification**: There are three people and three pets, creating multiple possible assignments. Without systematic reasoning, we cannot determine which person has which pet. The doubt arises because multiple arrangements are conceivable, and we must eliminate impossible ones to reach certainty.

---

## Pramana (Sources of Knowledge)

### Pratyaksha (Direct Perception)

```yaml
observable_facts:
  - "Alice does not have the cat"
  - "Bob has the dog"
  - "Carol does not have the fish"
  - "There are exactly three people: Alice, Bob, Carol"
  - "There are exactly three pets: cat, dog, fish"
  - "Each person has exactly one pet"
  - "Each pet belongs to exactly one person"
```

**Note**: These are the only directly stated facts. No inferences are included here.

### Anumana (Inference)

```yaml
inferences:
  - type: purvavat
    premise: "Bob has the dog (directly stated)"
    conclusion: "Neither Alice nor Carol has the dog"
    justification: "Since each pet belongs to exactly one person, if Bob has the dog, no one else can have it"
  
  - type: purvavat
    premise: "Alice does not have the cat, and Bob has the dog"
    conclusion: "Alice must have either the dog or the fish, but Bob has the dog, so Alice must have the fish"
    justification: "Alice has exactly one pet. She cannot have the cat (constraint 1) and cannot have the dog (Bob has it), leaving only the fish"
  
  - type: purvavat
    premise: "Alice has the fish, Bob has the dog"
    conclusion: "Carol must have the cat"
    justification: "All three pets must be assigned. Alice has fish, Bob has dog, leaving only cat for Carol"
```

### Upamana (Comparison)

```yaml
analogies:
  - reference: "Constraint satisfaction problems with mutual exclusivity"
    similarity: "This problem follows the same structure as assignment problems where each item in one set maps uniquely to one item in another set. The elimination method used here applies universally to such problems"
```

### Shabda (Testimony)

```yaml
principles:
  - "The Law of Excluded Middle: For any person and pet, either that person has that pet or they do not, with no third possibility"
  - "The Law of Non-Contradiction: No person can simultaneously have and not have the same pet"
  - "Mutual Exclusivity Principle: If one person has a pet, no other person can have that same pet"
  - "Completeness Principle: In a complete assignment problem, every person must have exactly one pet and every pet must belong to exactly one person"
```

---

## Pancha Avayava (5-Member Syllogism)

### Syllogism 1: Establishing Bob's Pet

**Pratijna (Thesis)**: Bob has the dog.

**Hetu (Reason)**: This is directly stated in constraint 2.

**Udaharana (Universal + Example)**: Wherever a constraint directly assigns a pet to a person, that assignment is true. For example, if we are told "X has Y," then X has Y. This is a universal principle of direct assignment in constraint satisfaction problems.

**Upanaya (Application)**: In this specific problem, constraint 2 states "Bob has the dog." This is a direct assignment, so the universal rule applies: Bob has the dog.

**Nigamana (Conclusion)**: Therefore, Bob has the dog.

### Syllogism 2: Establishing Alice's Pet

**Pratijna (Thesis)**: Alice has the fish.

**Hetu (Reason)**: Alice cannot have the cat (constraint 1), and she cannot have the dog (Bob has it from Syllogism 1). By the Law of Excluded Middle and mutual exclusivity, she must have the fish.

**Udaharana (Universal + Example)**: Wherever a person must have exactly one item from a set, and all but one possibility are eliminated, that person must have the remaining item. For example, if someone must choose one of three options (A, B, or C), and we know they cannot have A and cannot have B, then they must have C.

**Upanaya (Application)**: Alice must have exactly one pet from {cat, dog, fish}. She cannot have the cat (constraint 1). She cannot have the dog (Bob has it). Therefore, by elimination, she must have the fish.

**Nigamana (Conclusion)**: Therefore, Alice has the fish.

### Syllogism 3: Establishing Carol's Pet

**Pratijna (Thesis)**: Carol has the cat.

**Hetu (Reason)**: Alice has the fish (Syllogism 2), Bob has the dog (Syllogism 1), and Carol cannot have the fish (constraint 3). By completeness, Carol must have the cat.

**Udaharana (Universal + Example)**: Wherever all items in a set must be assigned and all but one assignment is determined, the remaining item must be assigned to the remaining person. For example, if we have three items {X, Y, Z} and three people {A, B, C}, and we know A has X and B has Y, then C must have Z.

**Upanaya (Application)**: We have three pets {cat, dog, fish} and three people {Alice, Bob, Carol}. Alice has the fish, Bob has the dog. Carol cannot have the fish (constraint 3), and she cannot have the dog (Bob has it). Therefore, Carol must have the cat.

**Nigamana (Conclusion)**: Therefore, Carol has the cat.

---

## Tarka (Counterfactual Reasoning)

**Hypothesis**: Suppose Carol does not have the cat (negation of our conclusion).

**Consequence**: If Carol does not have the cat, then Carol must have either the dog or the fish. However, Bob has the dog (established), so Carol cannot have the dog. Carol also cannot have the fish (constraint 3). Therefore, Carol would have no pet, which violates the completeness principle that every person must have exactly one pet.

**Analysis**: This leads to an absurdity: Carol would have no pet, yet the problem states each person has exactly one pet. This contradicts the fundamental structure of the problem.

**Resolution**: Therefore, our original conclusion must be true. Carol must have the cat.

**Additional Tarka Test**: Suppose Alice does not have the fish (negation of Syllogism 2). Then Alice must have either the cat or the dog. But Alice cannot have the cat (constraint 1), and Bob has the dog, so Alice cannot have the dog. This again leads to Alice having no pet, which is absurd. Therefore, Alice must have the fish.

---

## Hetvabhasa (Fallacy Check)

```yaml
fallacy_checks:
  savyabhichara: none_detected
    # Erratic/Inconclusive reasoning: Our reasoning is consistent. The Hetu (reasons) lead deterministically to the conclusions without contradiction.
  
  viruddha: none_detected
    # Contradictory reasoning: No contradictions exist. Our conclusions align with all constraints and do not contradict each other.
  
  prakaranasama: none_detected
    # Circular reasoning: We do not assume what we are trying to prove. We start from given constraints and derive conclusions through elimination, not by assuming the answer.
  
  sadhyasama: none_detected
    # Begging the question: Our premises (constraints) are independent of our conclusions. We do not use the conclusion to prove itself.
  
  kalaatita: none_detected
    # Temporal fallacy: No temporal reasoning is involved in this problem, so this fallacy type does not apply.

reasoning: "All reasoning steps follow valid logical principles. Each syllogism builds on previously established facts without circularity. The elimination method is applied systematically without assuming the conclusion. Tarka testing confirms the conclusions through reductio ad absurdum."
```

---

## Nirnaya (Ascertainment)

**Status**: Definitive Knowledge

**Final Answer**: Alice has the fish, Bob has the dog, and Carol has the cat.

**Justification**: All constraints are satisfied: Alice does not have the cat (she has the fish), Bob has the dog (directly stated), and Carol does not have the fish (she has the cat). The reasoning follows valid logical principles, all possibilities have been systematically eliminated, and Tarka testing confirms the solution through reductio ad absurdum. The answer is verifiable and complete.

**Confidence**: High - The solution is logically necessary given the constraints. No alternative assignment satisfies all constraints simultaneously.
