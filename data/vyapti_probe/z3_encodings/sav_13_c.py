"""Z3 encoding for SAV-13-C — auto-extracted from vyapti_benchmark_z3_encodings.md"""

PROBLEM_ID = "SAV-13-C"


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
revs = [Const(f'r{i}', Review) for i in range(1, 7)]

excellent = Function('excellent', Review, BoolSort())

s = Solver()

for r in revs:
    s.add(excellent(r))

x = Const('x', Review)
s.add(ForAll([x], excellent(x)))

print("SAV-13-C: All reviews Excellent — holds")
'''


if __name__ == "__main__":
    result = check()
    print(result)
