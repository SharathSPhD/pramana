"""Z3 encoding for PRA-02-C — auto-extracted from vyapti_benchmark_z3_encodings.md"""

PROBLEM_ID = "PRA-02-C"


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

Person = DeclareSort('Person')
kapoor = Const('kapoor', Person)

surgeon = Function('surgeon', Person, BoolSort())
completed_residency = Function('completed_residency', Person, BoolSort())

s = Solver()

x = Const('x', Person)
s.add(ForAll([x], Implies(surgeon(x), completed_residency(x))))
s.add(surgeon(kapoor))

s.push()
s.add(Not(completed_residency(kapoor)))
print(f"PRA-02-C: Can Kapoor NOT have completed residency? {s.check() == sat}")  # UNSAT
s.pop()

# Correct answer: Same — completed residency
'''


if __name__ == "__main__":
    result = check()
    print(result)
