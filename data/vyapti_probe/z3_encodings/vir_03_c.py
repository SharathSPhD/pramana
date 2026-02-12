"""Z3 encoding for VIR-03-C — auto-extracted from vyapti_benchmark_z3_encodings.md"""

PROBLEM_ID = "VIR-03-C"


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

scored_above_90 = Bool('scored_above_90')
scholarship = Bool('scholarship')

s = Solver()
s.add(Implies(scored_above_90, scholarship))
s.add(scored_above_90)  # Fact: scored 95

s.push()
s.add(Not(scholarship))
print(f"VIR-03-C: Can student NOT have scholarship? {s.check() == sat}")  # UNSAT
s.pop()

# Correct answer: Yes (scholarship)
'''


if __name__ == "__main__":
    result = check()
    print(result)
