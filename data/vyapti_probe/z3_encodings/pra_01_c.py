"""Z3 encoding for PRA-01-C — auto-extracted from vyapti_benchmark_z3_encodings.md"""

PROBLEM_ID = "PRA-01-C"


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

Amit, Bela, Charu, Dev, Ela = Ints('Amit Bela Charu Dev Ela')

s = Solver()

for p in [Amit, Bela, Charu, Dev, Ela]:
    s.add(And(p >= 1, p <= 5))
s.add(Distinct(Amit, Bela, Charu, Dev, Ela))

s.add(Amit == 1)
s.add(Bela == Amit + 1)
s.add(Charu == 5)
s.add(Ela != 3)  # Added constraint

# Ela's position now determined: 4
s.push()
s.add(Ela == 4)
r = s.check()
s.pop()

s.push()
s.add(Ela != 4)
r2 = s.check()
s.pop()

print(f"PRA-01-C: Ela=4 forced: {r == sat and r2 == unsat}")
'''


if __name__ == "__main__":
    result = check()
    print(result)
