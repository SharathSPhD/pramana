"""Z3 encoding for VIR-03 — auto-extracted from vyapti_benchmark_z3_encodings.md"""

PROBLEM_ID = "VIR-03"


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

# Rule: scored > 90 → scholarship
s.add(Implies(scored_above_90, scholarship))

# Fact: scholarship received
s.add(scholarship)

# Check: is scored_above_90 forced?
s.push()
s.add(Not(scored_above_90))
print(f"VIR-03: Can student NOT have scored above 90? {s.check() == sat}")  # SAT — not forced
s.pop()
# Correct: cannot conclude scored > 90
# Trap: "yes, scored > 90"
'''


if __name__ == "__main__":
    result = check()
    print(result)
