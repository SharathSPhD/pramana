"""Z3 encoding for PRA-01 — auto-extracted from vyapti_benchmark_z3_encodings.md"""

PROBLEM_ID = "PRA-01"


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

# Positions 1-5 for 5 people
Amit, Bela, Charu, Dev, Ela = Ints('Amit Bela Charu Dev Ela')

s = Solver()

for p in [Amit, Bela, Charu, Dev, Ela]:
    s.add(And(p >= 1, p <= 5))
s.add(Distinct(Amit, Bela, Charu, Dev, Ela))

s.add(Amit == 1)       # Constraint 1
s.add(Bela == Amit + 1) # Constraint 2 → Bela = 2
# Constraint 3 (Dev prefers coffee) is NOT encodable — it's irrelevant
s.add(Charu == 5)       # Constraint 4

# Check: is Ela's position determined?
s.push()
s.add(Ela == 3)
r1 = s.check()
s.pop()

s.push()
s.add(Ela == 4)
r2 = s.check()
s.pop()

print(f"PRA-01: Ela can be 3: {r1 == sat}, Ela can be 4: {r2 == sat}")
# Both SAT — Ela's position is NOT determined
# Dev's coffee preference has zero bearing on this
'''


if __name__ == "__main__":
    result = check()
    print(result)
