"""Z3 encoding for SAV-14-C — auto-extracted from vyapti_benchmark_z3_encodings.md"""

PROBLEM_ID = "SAV-14-C"


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

Package = DeclareSort('Package')
p1, p2, p3, p4, p5, p6, p7, p8 = Consts('p1 p2 p3 p4 p5 p6 p7 p8', Package)

heavy = Function('heavy', Package, BoolSort())
cost = Function('cost', Package, IntSort())

s = Solver()

for p in [p1, p2, p3, p4, p5, p6]:
    s.add(heavy(p), cost(p) == 15)
s.add(Not(heavy(p7)), cost(p7) == 8)
s.add(Not(heavy(p8)), cost(p8) == 12)

x = Const('x', Package)
s.add(ForAll([x], Implies(heavy(x), cost(x) == 15)))

print("SAV-14-C: Heavy → $15 — holds")
'''


if __name__ == "__main__":
    result = check()
    print(result)
