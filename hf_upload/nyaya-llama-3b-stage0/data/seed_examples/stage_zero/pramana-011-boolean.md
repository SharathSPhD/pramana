---
id: pramana-011
problem_type: boolean_sat
difficulty: medium
variables: 3
ground_truth: "A is a Knave, B is a Knave, C is a Knight"
metadata:
  stage: 0
  verified: true
  created_at: 2026-01-31
  z3_verifiable: true
---

# Problem

On an island, there are three inhabitants: A, B, and C. Each person is either a Knight (always tells the truth) or a Knave (always lies). You encounter them and hear the following statements:

**Statements**:
1. A says: "B is a Knight."
2. B says: "A and C are the same type."
3. C says: "A is a Knave."

**Question**: What type is each person (Knight or Knave)?

---

## Samshaya (Doubt Analysis)

**Doubt Type**: Viparyaya Samshaya (Doubt arising from contradictory possibilities)

**Justification**: Each person can be either a Knight or a Knave, creating 2³ = 8 possible assignments. However, the statements create logical dependencies where the truth value of each statement depends on the type of speaker. Statement 2 introduces a compound condition about A and C being the same type, which adds complexity to the logical constraints that must be simultaneously satisfied.

---

## Pramana (Sources of Knowledge)

### Pratyaksha (Direct Perception)

```yaml
observable_facts:
  - "A said: 'B is a Knight'"
  - "B said: 'A and C are the same type'"
  - "C said: 'A is a Knave'"
  - "There are three people: A, B, and C"
  - "Each person is either a Knight or a Knave (no other possibilities)"
  - "Knights always tell the truth"
  - "Knaves always lie"
  - "'A and C are the same type' means: (A is Knight and C is Knight) or (A is Knave and C is Knave)"
```

**Note**: We observe only what was said, not the truth value of the statements. The truth value depends on the type of speaker.

### Anumana (Inference)

```yaml
inferences:
  - type: purvavat
    premise: "If A is a Knight, then A's statement 'B is a Knight' is true, so B is a Knight"
    conclusion: "A is Knight → B is Knight"
    justification: "Knights tell truth, so their statements are true"
  
  - type: purvavat
    premise: "If A is a Knave, then A's statement 'B is a Knight' is false, so B is not a Knight, meaning B is a Knave"
    conclusion: "A is Knave → B is Knave"
    justification: "Knaves lie, so their statements are false"
  
  - type: purvavat
    premise: "If B is a Knight, then B's statement 'A and C are the same type' is true, so (A is Knight and C is Knight) or (A is Knave and C is Knave)"
    conclusion: "B is Knight → (A and C same type)"
    justification: "Knights tell truth"
  
  - type: purvavat
    premise: "If B is a Knave, then B's statement 'A and C are the same type' is false, so A and C are different types"
    conclusion: "B is Knave → (A and C different types)"
    justification: "Knaves lie"
  
  - type: purvavat
    premise: "If C is a Knight, then C's statement 'A is a Knave' is true, so A is a Knave"
    conclusion: "C is Knight → A is Knave"
    justification: "Knights tell truth"
  
  - type: purvavat
    premise: "If C is a Knave, then C's statement 'A is a Knave' is false, so A is not a Knave, meaning A is a Knight"
    conclusion: "C is Knave → A is Knight"
    justification: "Knaves lie"
```

### Upamana (Comparison)

```yaml
analogies:
  - reference: "Boolean satisfiability problems with compound logical conditions"
    similarity: "This problem maps to propositional logic with compound statements. The statement 'A and C are the same type' creates a logical equivalence: (A is Knight) ↔ (C is Knight). The problem structure is isomorphic to finding a satisfying assignment for a system of logical constraints including compound conditions"
```

### Shabda (Testimony)

```yaml
principles:
  - "Law of Excluded Middle: Each person is either a Knight or a Knave, with no third possibility"
  - "Principle of Truth-Telling: If X is a Knight, then every statement X makes is true"
  - "Principle of Lying: If X is a Knave, then every statement X makes is false"
  - "Logical Equivalence: 'X says Y' means: (X is Knight) ↔ Y"
  - "Same Type Principle: 'A and C are the same type' means: (A is Knight) ↔ (C is Knight)"
  - "Consistency Principle: All logical implications must be simultaneously satisfiable for a valid solution"
```

---

## Pancha Avayava (5-Member Syllogism)

### Syllogism 1: Establishing the Logical Structure

**Pratijna (Thesis)**: The statements create a system of logical constraints that must be simultaneously satisfied.

**Hetu (Reason)**: Each statement "X says Y" means: (X is Knight) ↔ Y. Statement 2 introduces a compound condition about A and C being the same type.

**Udaharana (Universal + Example)**: Wherever we have statements of the form "X says Y" in a Knights and Knaves problem, the logical structure is: (X is Knight) ↔ Y. For example, if we have "Alice says Bob is a Knight," this means: (Alice is Knight) ↔ (Bob is Knight). This universal rule applies to all such problems.

**Upanaya (Application)**: In this problem, A says "B is a Knight" means: (A is Knight) ↔ (B is Knight). B says "A and C are the same type" means: (B is Knight) ↔ ((A is Knight) ↔ (C is Knight)). C says "A is a Knave" means: (C is Knight) ↔ ¬(A is Knight), or equivalently: (C is Knight) ↔ (A is Knave). These three constraints must all hold simultaneously.

**Nigamana (Conclusion)**: Therefore, we have a system of logical constraints that must be satisfied: (A is Knight) ↔ (B is Knight), (B is Knight) ↔ ((A is Knight) ↔ (C is Knight)), (C is Knight) ↔ (A is Knave).

### Syllogism 2: Deriving the Solution Through Systematic Testing

**Pratijna (Thesis)**: A is a Knave, B is a Knave, and C is a Knight.

**Hetu (Reason)**: This assignment satisfies all logical constraints simultaneously when verified through systematic checking.

**Udaharana (Universal + Example)**: Wherever we systematically test assignments in a Knights and Knaves problem, we verify that each person's statement is consistent with their type. For example, if we test "X is Knave, Y is Knave, Z is Knight," we check: X (Knave) lies, Y (Knave) lies, Z (Knight) tells truth, and verify each statement matches these expectations.

**Upanaya (Application)**: Testing A=Knave, B=Knave, C=Knight: A (Knave) says "B is Knight" → False (A lies) → B is not Knight, meaning B is Knave ✓. B (Knave) says "A and C are the same type" → False (B lies) → A and C are not the same type. A is Knave and C is Knight, so they are different types ✓. C (Knight) says "A is Knave" → True (C tells truth) → A is Knave ✓. All statements are consistent with this assignment.

**Nigamana (Conclusion)**: Therefore, A is a Knave, B is a Knave, and C is a Knight.

---

## Tarka (Counterfactual Reasoning)

**Hypothesis**: Suppose A is not a Knave (i.e., A is a Knight).

**Consequence**: If A is a Knight, then A's statement "B is a Knight" is true, so B is a Knight. If B is a Knight, then B's statement "A and C are the same type" is true, so A and C are the same type. Since A is a Knight, C must also be a Knight. If C is a Knight, then C's statement "A is a Knave" is true, so A is a Knave. But we assumed A is a Knight, which is a contradiction.

**Analysis**: The hypothesis leads to a logical contradiction. We cannot have A as Knight while satisfying all statements simultaneously.

**Resolution**: Therefore, A must be a Knave. Given A is a Knave, A's statement "B is a Knight" is false, so B is a Knave. Given B is a Knave, B's statement "A and C are the same type" is false, so A and C are different types. Since A is a Knave, C must be a Knight. Given C is a Knight, C's statement "A is a Knave" is true, confirming A is a Knave. This assignment is consistent.

---

## Hetvabhasa (Fallacy Check)

```yaml
fallacy_checks:
  savyabhichara: none_detected
    # Erratic reasoning: Our systematic testing of assignments is consistent and methodical. We check each possibility without contradiction.
  
  viruddha: none_detected
    # Contradictory reasoning: No contradictions exist in our logical chain when we follow the correct assignment. All statements are consistent.
  
  prakaranasama: none_detected
    # Circular reasoning: We do not assume the conclusion. We test assignments and verify consistency, building from the given statements.
  
  sadhyasama: none_detected
    # Begging the question: Our reasoning starts from the given statements and derives the solution through systematic verification, not by assuming the answer.
  
  kalaatita: none_detected
    # Temporal fallacy: No temporal reasoning involved in this problem.

reasoning: "We systematically test each possible assignment and verify consistency with all statements. The reasoning follows logical principles without circularity or contradiction. Each step builds from the given constraints without assuming the conclusion."
```

---

## Nirnaya (Ascertainment)

**Status**: Definitive Knowledge

**Final Answer**: A is a Knave, B is a Knave, and C is a Knight.

**Justification**: This assignment satisfies all statements: A (Knave) lies that B is Knight (so B is Knave), B (Knave) lies that A and C are the same type (so A and C are different types, which is true since A is Knave and C is Knight), and C (Knight) tells truth that A is Knave. The solution is verified through systematic testing and Tarka counterfactual analysis.

**Confidence**: High - The assignment is logically consistent with all given statements and can be verified through direct checking. The solution satisfies all logical constraints.