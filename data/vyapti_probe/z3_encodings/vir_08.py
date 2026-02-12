"""Z3 encoding for VIR-08 — auto-extracted from vyapti_benchmark_z3_encodings.md"""

PROBLEM_ID = "VIR-08"


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

temp_under_60 = Bool('temp_under_60')
heater_on = Bool('heater_on')

s = Solver()

# Rule: temp<60 → heater ON (conditional, not biconditional)
s.add(Implies(temp_under_60, heater_on))

# Fact: heater ON
s.add(heater_on)

# Check: is temp<60 forced? (converse — invalid)
s.push()
s.add(Not(temp_under_60))
print(f"VIR-08: Can temp be ≥60? {s.check() == sat}")  # SAT — cannot conclude
s.pop()

# Correct answer: Cannot conclude (converse fallacy)
'''


if __name__ == "__main__":
    result = check()
    print(result)
