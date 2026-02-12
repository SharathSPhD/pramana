"""Z3 encoding for VIR-04-C — auto-extracted from vyapti_benchmark_z3_encodings.md"""

PROBLEM_ID = "VIR-04-C"


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
s.add(Implies(Not(training), Not(permitted)))  # permitted → training
s.add(permitted)  # Fact: inside zone

# By contrapositive: training must be completed
s.push()
s.add(Not(training))
print(f"VIR-04-C: Can training NOT be completed? {s.check() == sat}")  # UNSAT
s.pop()

# Correct answer: Yes (training completed)
'''


if __name__ == "__main__":
    result = check()
    print(result)
