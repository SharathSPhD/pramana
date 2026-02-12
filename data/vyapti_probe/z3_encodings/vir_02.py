"""Z3 encoding for VIR-02 — auto-extracted from vyapti_benchmark_z3_encodings.md"""

PROBLEM_ID = "VIR-02"


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

test_a = Bool('test_a')
test_b = Bool('test_b')
rejected = Bool('rejected')

s = Solver()

# Rule: rejected ↔ ¬(test_a ∧ test_b)
s.add(rejected == Not(And(test_a, test_b)))

# Fact: test_a passed, test_b failed
s.add(test_a)
s.add(Not(test_b))

# Check: is rejected forced?
s.push()
s.add(Not(rejected))
print(f"VIR-02: Can product NOT be rejected? {s.check() == sat}")  # UNSAT — rejected is forced
s.pop()

# Correct answer: Yes (rejected)
'''


if __name__ == "__main__":
    result = check()
    print(result)
