"""Z3 encoding for VIR-10 — auto-extracted from vyapti_benchmark_z3_encodings.md"""

PROBLEM_ID = "VIR-10"


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

A = Bool('A')
B = Bool('B')
prize = Bool('prize')

s = Solver()

# Rule: A∨B → prize (inclusive or)
s.add(Implies(Or(A, B), prize))

# Fact: A∧B
s.add(A)
s.add(B)

# Check: is prize forced?
s.push()
s.add(Not(prize))
print(f"VIR-10: Can prize be denied? {s.check() == sat}")  # UNSAT
s.pop()

# Correct answer: Yes (prize)
'''


if __name__ == "__main__":
    result = check()
    print(result)
