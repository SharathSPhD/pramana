"""Z3 encoding for SAV-10 — auto-extracted from vyapti_benchmark_z3_encodings.md"""

PROBLEM_ID = "SAV-10"


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

# Routes: 4 factories. Factory D: 13 days; others: 9 days
Route = DeclareSort('Route')
r1, r2, r3, r4 = Consts('r1 r2 r3 r4', Route)
days = Function('days', Route, IntSort())

s = Solver()

s.add(days(r1) == 9)
s.add(days(r2) == 9)
s.add(days(r3) == 9)
s.add(days(r4) == 13)  # Factory D route = 13 days

# Vyāpti: all routes 9 days — r4 contradicts
s.push()
s.add(days(r4) == 9)
print(f"SAV-10: All routes 9 days? {s.check() == sat}")  # UNSAT — r4=13
s.pop()

# Correct answer: No (Factory D = 13 days)
'''


if __name__ == "__main__":
    result = check()
    print(result)
