"""Z3 encoding for SAD-01 — auto-extracted from vyapti_benchmark_z3_encodings.md"""

PROBLEM_ID = "SAD-01"


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

# A defined by B, B defined by A. Q: Does X have property B? A: Cannot determine (circular)

# Circular: has_A(x) ↔ has_B(x), has_B(x) ↔ has_A(x)
# No independent anchor — value of B for X is underdetermined

Entity = DeclareSort('Entity')
x = Const('x', Entity)
has_A = Function('has_A', Entity, BoolSort())
has_B = Function('has_B', Entity, BoolSort())

s = Solver()

# Circular: A iff B
s.add(ForAll([x], has_A(x) == has_B(x)))
s.add(ForAll([x], has_B(x) == has_A(x)))

# Check: is has_B(x) determined?
s.push()
s.add(has_B(x))
r1 = s.check()
s.pop()

s.push()
s.add(Not(has_B(x)))
r2 = s.check()
s.pop()

print(f"SAD-01: B can be True: {r1 == sat}, B can be False: {r2 == sat}")
# Both SAT — circular definition, cannot determine
'''


if __name__ == "__main__":
    result = check()
    print(result)
