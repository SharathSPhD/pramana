"""Z3 encoding for VIR-09 — auto-extracted from vyapti_benchmark_z3_encodings.md"""

PROBLEM_ID = "VIR-09"


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

high_priority = Bool('high_priority')
senior_partner = Bool('senior_partner')

s = Solver()

# Rule: high_priority → senior_partner
s.add(Implies(high_priority, senior_partner))

# Fact: NOT high_priority
s.add(Not(high_priority))

# Check: is senior_partner forced to false?
s.push()
s.add(senior_partner)
print(f"VIR-09: Can senior partner be assigned? {s.check() == sat}")  # SAT — not forced false
s.pop()
# Correct: cannot conclude ¬senior_partner
# Trap: "no senior partner" (denying antecedent fallacy)
'''


if __name__ == "__main__":
    result = check()
    print(result)
