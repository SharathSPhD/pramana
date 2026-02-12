"""Z3 encoding for VIR-07-C — auto-extracted from vyapti_benchmark_z3_encodings.md"""

PROBLEM_ID = "VIR-07-C"


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

overloaded = Bool('overloaded')
network_down = Bool('network_down')
backup = Bool('backup')

s = Solver()
s.add(Implies(Or(overloaded, network_down), backup))
s.add(overloaded)  # Fact: overloaded

s.push()
s.add(Not(backup))
print(f"VIR-07-C: Can backup NOT activate? {s.check() == sat}")  # UNSAT
s.pop()

# Correct answer: Yes (backup activates)
'''


if __name__ == "__main__":
    result = check()
    print(result)
