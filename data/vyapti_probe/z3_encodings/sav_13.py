"""Z3 encoding for SAV-13 — auto-extracted from vyapti_benchmark_z3_encodings.md"""

PROBLEM_ID = "SAV-13"


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

Review = DeclareSort('Review')
r1, r2, r3, r4, r5, r6 = Consts('r1 r2 r3 r4 r5 r6', Review)

visit = Function('visit', Review, BoolSort())  # Any visit (lunch or dinner)
excellent = Function('excellent', Review, BoolSort())

s = Solver()

# Reviews 1-3, 5-6: Lunch, Excellent
s.add(visit(r1), excellent(r1))
s.add(visit(r2), excellent(r2))
s.add(visit(r3), excellent(r3))
s.add(visit(r4), Not(excellent(r4)))  # Review 4: Dinner (visit), Poor
s.add(visit(r5), excellent(r5))
s.add(visit(r6), excellent(r6))

# Vyāpti: visit → excellent. Review 4 is a visit, not excellent.
s.push()
s.add(visit(r4), Not(excellent(r4)))
print(f"SAV-13: 'Visit → Excellent' holds: {s.check() != sat}")  # False
s.pop()

# Correct answer: No (Review 4)
'''


if __name__ == "__main__":
    result = check()
    print(result)
