"""Z3 encoding for PRA-03 — auto-extracted from vyapti_benchmark_z3_encodings.md"""

PROBLEM_ID = "PRA-03"


def check():
    """Run the encoding, capture print output, return dict with output and problem_id."""
    import sys
    from io import StringIO

    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        exec(compile(_CODE, "<encoding>", "exec"))
        output = sys.stdout.getvalue()
    finally:
        sys.stdout = old_stdout
    return {"output": output, "problem_id": PROBLEM_ID}


_CODE = '''
from z3 import *

X, Y, Z, Q = Bools('X Y Z Q')

# Rules: X,Y,Z→Q. Rule 3: Z∨¬Z (tautology). Given X=T,Y=T,Z=F.
# Q? A: True (Z is irrelevant when Z=F)

s = Solver()

# Simplified: (X ∧ Y) → Q, with X=T, Y=T
s.add(Implies(And(X, Y), Q))
s.add(X == True)
s.add(Y == True)
s.add(Z == False)

# Z∨¬Z is always true — doesn't constrain
s.add(Or(Z, Not(Z)))

s.push()
s.add(Not(Q))
print(f"PRA-03: Can Q be false? {s.check() == sat}")  # UNSAT
s.pop()

# Correct answer: Q = True
'''


if __name__ == "__main__":
    result = check()
    print(result)
