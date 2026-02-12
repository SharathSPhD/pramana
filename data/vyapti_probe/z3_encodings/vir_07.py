"""Z3 encoding for VIR-07 — auto-extracted from vyapti_benchmark_z3_encodings.md"""

PROBLEM_ID = "VIR-07"


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

# Rule: (overloaded ∨ network_down) → backup
s.add(Implies(Or(overloaded, network_down), backup))

# Fact: backup NOT activated
s.add(Not(backup))

# Check: are BOTH overloaded and network_down forced to false?
s.push()
s.add(overloaded)
print(f"VIR-07: Can server be overloaded? {s.check() == sat}")  # UNSAT
s.pop()

s.push()
s.add(network_down)
print(f"VIR-07: Can network be down? {s.check() == sat}")  # UNSAT
s.pop()

# Correct: ¬overloaded ∧ ¬network_down
# Trap: "one of them might still be true"
'''


if __name__ == "__main__":
    result = check()
    print(result)
