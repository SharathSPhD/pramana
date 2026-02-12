"""Z3 encoding for SAD-01-C — auto-extracted from vyapti_benchmark_z3_encodings.md"""

PROBLEM_ID = "SAD-01-C"


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
x = Const('x', Entity)
has_B = Function('has_B', Entity, BoolSort())
molecular_weight = Function('molecular_weight', Entity, IntSort())

s = Solver()

# B = molecular weight > 500 (independent definition)
s.add(ForAll([x], has_B(x) == (molecular_weight(x) > 500)))
s.add(molecular_weight(x) == 750)

s.push()
s.add(Not(has_B(x)))
print(f"SAD-01-C: Can X lack B? {s.check() == sat}")  # UNSAT
s.pop()

print("SAD-01-C: Yes — X has B (MW 750 > 500)")
'''


if __name__ == "__main__":
    result = check()
    print(result)
