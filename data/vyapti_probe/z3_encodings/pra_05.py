"""Z3 encoding for PRA-05 — auto-extracted from vyapti_benchmark_z3_encodings.md"""

PROBLEM_ID = "PRA-05"


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

# Lab assignment: 0 = North, 1 = South
Priya, Qadir, Ravi, Sita = Ints('Priya Qadir Ravi Sita')

s = Solver()

for p in [Priya, Qadir, Ravi, Sita]:
    s.add(Or(p == 0, p == 1))  # 0=North, 1=South

s.add(Priya == 0)   # Constraint 1
s.add(Qadir == 1)    # Constraint 2
# Constraint 3 (Ravi drinks coffee) — NOT encodable, irrelevant
s.add(Sita == 0)     # Constraint 4
# Constraint 5 (Priya drinks tea) — NOT encodable, irrelevant

# Check: is Ravi's lab determined?
s.push()
s.add(Ravi == 0)
r1 = s.check()
s.pop()

s.push()
s.add(Ravi == 1)
r2 = s.check()
s.pop()

print(f"PRA-05: Ravi can be North: {r1 == sat}, South: {r2 == sat}")
# Both SAT — beverage preferences don't constrain lab assignment
'''


if __name__ == "__main__":
    result = check()
    print(result)
