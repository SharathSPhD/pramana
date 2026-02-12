"""Z3 encoding for SAD-06-C — auto-extracted from vyapti_benchmark_z3_encodings.md"""

PROBLEM_ID = "SAD-06-C"


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

Entity = DeclareSort('Entity')
q = Const('q', Entity)
type_x = Function('type_x', Entity, BoolSort())
temperature = Function('temperature', Entity, IntSort())

s = Solver()

# Feature-P = temperature > 1000 (independent)
s.add(ForAll([q], type_x(q) == (temperature(q) > 1000)))
s.add(temperature(q) == 1200)

s.push()
s.add(Not(type_x(q)))
print(f"SAD-06-C: Can Q lack type_x? {s.check() == sat}")  # UNSAT
s.pop()

print("SAD-06-C: Yes — valid classification")
'''


if __name__ == "__main__":
    result = check()
    print(result)
