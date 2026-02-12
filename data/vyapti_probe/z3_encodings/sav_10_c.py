"""Z3 encoding for SAV-10-C — auto-extracted from vyapti_benchmark_z3_encodings.md"""

PROBLEM_ID = "SAV-10-C"


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

Route = DeclareSort('Route')
r1, r2, r3, r4 = Consts('r1 r2 r3 r4', Route)
days = Function('days', Route, IntSort())

s = Solver()

s.add(days(r1) == 9)
s.add(days(r2) == 9)
s.add(days(r3) == 9)
s.add(days(r4) == 9)  # All uniform

s.check()
m = s.model()
print(f"SAV-10-C: All routes 9 days: {m.eval(days(r4)) == 9}")

# Correct answer: Yes
'''


if __name__ == "__main__":
    result = check()
    print(result)
