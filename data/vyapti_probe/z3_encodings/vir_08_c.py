"""Z3 encoding for VIR-08-C — auto-extracted from vyapti_benchmark_z3_encodings.md"""

PROBLEM_ID = "VIR-08-C"


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
s.add(Implies(temp_under_60, heater_on))
s.add(temp_under_60)  # Fact: temp=55

s.push()
s.add(Not(heater_on))
print(f"VIR-08-C: Can heater be OFF? {s.check() == sat}")  # UNSAT
s.pop()

# Correct answer: Yes (heater ON)
'''


if __name__ == "__main__":
    result = check()
    print(result)
