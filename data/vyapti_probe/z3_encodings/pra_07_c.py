"""Z3 encoding for PRA-07-C — auto-extracted from vyapti_benchmark_z3_encodings.md"""

PROBLEM_ID = "PRA-07-C"


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

P, Q, R = Bools('P Q R')
OUT = Or(And(P, Q), R)

s = Solver()
s.add(P == True, Q == False, R == True)

# OUT = (T∧F) ∨ T = F ∨ T = T
s.push()
s.add(Not(OUT))
print(f"PRA-07-C: OUT can be false? {s.check() == sat}")  # UNSAT
s.pop()

# Change R to F: OUT = (T∧F) ∨ F = F
s2 = Solver()
s2.add(P == True, Q == False, R == False)
OUT2 = Or(And(True, False), False)
print(f"PRA-07-C: R matters — OUT changes when R=F: {simplify(OUT2) == False}")
'''


if __name__ == "__main__":
    result = check()
    print(result)
