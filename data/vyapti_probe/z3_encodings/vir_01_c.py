"""Z3 encoding for VIR-01-C — auto-extracted from vyapti_benchmark_z3_encodings.md"""

PROBLEM_ID = "VIR-01-C"


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
s.add(Implies(passed, homework))
s.add(passed)  # Student DID pass

s.push()
s.add(Not(homework))
print(f"VIR-01-C: Can student NOT have done homework? {s.check() == sat}")  # UNSAT
s.pop()
# Correct: homework must be true
'''


if __name__ == "__main__":
    result = check()
    print(result)
