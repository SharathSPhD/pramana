"""Z3 encoding for VIR-01 — auto-extracted from vyapti_benchmark_z3_encodings.md"""

PROBLEM_ID = "VIR-01"


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

passed = Bool('passed')
homework = Bool('homework')

s = Solver()

# Rule: passed → homework
s.add(Implies(passed, homework))

# Fact: NOT homework
s.add(Not(homework))

# Check: is passed forced to be false?
s.push()
s.add(passed)
print(f"VIR-01: Can student have passed? {s.check() == sat}")  # UNSAT — modus tollens forces ¬passed
s.pop()

# Correct: ¬passed (by contrapositive)
# Trap: "can't determine" or "might have passed"
'''


if __name__ == "__main__":
    result = check()
    print(result)
