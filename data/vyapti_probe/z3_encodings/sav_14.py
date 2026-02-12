"""Z3 encoding for SAV-14 — auto-extracted from vyapti_benchmark_z3_encodings.md"""

PROBLEM_ID = "SAV-14"


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
international = Function('international', Package, BoolSort())
cost = Function('cost', Package, IntSort())

s = Solver()

for p in [p1, p2, p3, p4, p5]:
    s.add(heavy(p), cost(p) == 15)
s.add(heavy(p6), international(p6), cost(p6) == 25)  # p6: Heavy+Intl = $25
s.add(Not(heavy(p7)), cost(p7) == 8)
s.add(Not(heavy(p8)), cost(p8) == 12)

# Vyāpti: heavy → $15
s.push()
s.add(heavy(p6), cost(p6) != 15)
print(f"SAV-14: 'Heavy → $15' holds: {s.check() != sat}")  # False
s.pop()

# Correct answer: No (Package 6)
'''


if __name__ == "__main__":
    result = check()
    print(result)
