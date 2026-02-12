"""Z3 encoding for PRA-06 — auto-extracted from vyapti_benchmark_z3_encodings.md"""

PROBLEM_ID = "PRA-06"


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

# 1) Chess→chess. 2) Rishi∈chess. 3) Rishi owns bicycle. 4) Bicycle→within 5km.
# Q: Rishi within 5km? A: Yes (via 3+4, not 1+2)

Person = DeclareSort('Person')
rishi = Const('rishi', Person)

chess_club = Function('chess_club', Person, BoolSort())
owns_bicycle = Function('owns_bicycle', Person, BoolSort())
within_5km = Function('within_5km', Person, BoolSort())

s = Solver()

# Premise 3+4: owns_bicycle → within_5km
x = Const('x', Person)
s.add(ForAll([x], Implies(owns_bicycle(x), within_5km(x))))
s.add(owns_bicycle(rishi))

s.push()
s.add(Not(within_5km(rishi)))
print(f"PRA-06: Can Rishi be outside 5km? {s.check() == sat}")  # UNSAT
s.pop()

# Correct answer: Yes (via 3+4)
print("PRA-06: Rishi within 5km (premises 1+2 are red herring)")
'''


if __name__ == "__main__":
    result = check()
    print(result)
