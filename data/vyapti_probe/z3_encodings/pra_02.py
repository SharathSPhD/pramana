"""Z3 encoding for PRA-02 — auto-extracted from vyapti_benchmark_z3_encodings.md"""

PROBLEM_ID = "PRA-02"


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

# 1) All surgeons completed residency
# 2) Dr. Kapoor is surgeon
# 3) Cafeteria serves vegetarian Fridays (irrelevant)
# 4) All housing residents drive sedans (irrelevant)
# Q: Dr. Kapoor? A: Completed residency only

surgeon = Function('surgeon', DeclareSort('Person'), BoolSort())
completed_residency = Function('completed_residency', DeclareSort('Person'), BoolSort())

Person = DeclareSort('Person')
kapoor = Const('kapoor', Person)

s = Solver()

# Premises 1 and 2
x = Const('x', Person)
s.add(ForAll([x], Implies(surgeon(x), completed_residency(x))))
s.add(surgeon(kapoor))

# Premises 3, 4 — not encoded (irrelevant)

s.push()
s.add(Not(completed_residency(kapoor)))
print(f"PRA-02: Can Kapoor NOT have completed residency? {s.check() == sat}")  # UNSAT
s.pop()

# Correct answer: Completed residency only
print("PRA-02: Kapoor completed residency (premises 3,4 irrelevant)")
'''


if __name__ == "__main__":
    result = check()
    print(result)
