"""Z3 encoding for VIR-04 — auto-extracted from vyapti_benchmark_z3_encodings.md"""

PROBLEM_ID = "VIR-04"


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

training = Bool('training')
permitted = Bool('permitted')

s = Solver()

# Rule: ¬training → ¬permitted (equivalently: permitted → training)
s.add(Implies(Not(training), Not(permitted)))

# Fact: training completed
s.add(training)

# Check: is permitted forced?
s.push()
s.add(Not(permitted))
print(f"VIR-04: Can employee NOT be permitted? {s.check() == sat}")  # SAT — training is necessary but not sufficient
s.pop()
# Correct: cannot conclude permitted
# Trap: "yes, permitted"
'''


if __name__ == "__main__":
    result = check()
    print(result)
