"""Z3 encoding for PRA-06-C — auto-extracted from vyapti_benchmark_z3_encodings.md"""

PROBLEM_ID = "PRA-06-C"


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
rishi = Const('rishi', Person)

owns_bicycle = Function('owns_bicycle', Person, BoolSort())
within_5km = Function('within_5km', Person, BoolSort())

s = Solver()

x = Const('x', Person)
s.add(ForAll([x], Implies(owns_bicycle(x), within_5km(x))))
s.add(owns_bicycle(rishi))

s.push()
s.add(Not(within_5km(rishi)))
print(f"PRA-06-C: Can Rishi be outside 5km? {s.check() == sat}")  # UNSAT
s.pop()

# Correct answer: Yes
'''


if __name__ == "__main__":
    result = check()
    print(result)
