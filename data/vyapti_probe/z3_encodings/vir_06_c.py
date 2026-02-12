"""Z3 encoding for VIR-06-C — auto-extracted from vyapti_benchmark_z3_encodings.md"""

PROBLEM_ID = "VIR-06-C"


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

Light = DeclareSort('Light')
l1, l2, l3 = Consts('l1 l2 l3', Light)
on = Function('on', Light, BoolSort())

s = Solver()

# Premise: All lights in Room A are on
s.add(on(l1), on(l2), on(l3))

# What must be true? (c) Both: at least one on, none off
s.push()
s.add(Not(on(l1)))
print(f"VIR-06-C: Can any light be off? {s.check() == sat}")  # UNSAT
s.pop()

# Correct answer: (c) Both (a) and (b)
print("VIR-06-C: All on → at least one on AND none off")
'''


if __name__ == "__main__":
    result = check()
    print(result)
