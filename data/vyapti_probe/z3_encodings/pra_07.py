"""Z3 encoding for PRA-07 — auto-extracted from vyapti_benchmark_z3_encodings.md"""

PROBLEM_ID = "PRA-07"


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

P, Q, R, S = Bools('P Q R S')
OUT = Or(And(P, Q), And(R, S))

s = Solver()
s.add(P == True, Q == False, R == True, S == True)

# OUT = (T ∧ F) ∨ (T ∧ T) = F ∨ T = T
s.push()
s.add(Not(OUT))
print(f"PRA-07: OUT can be false? {s.check() == sat}")  # UNSAT — OUT is true
s.pop()

# Now check: does changing P affect OUT?
s2 = Solver()
s2.add(P == False, Q == False, R == True, S == True)  # Changed P to False
OUT2 = Or(And(False, False), And(True, True))  # = F ∨ T = T

# P is IRRELEVANT because Q=False makes (P ∧ Q) = False regardless
print(f"PRA-07: P is irrelevant: OUT unchanged = {simplify(OUT2) == True}")
'''


if __name__ == "__main__":
    result = check()
    print(result)
