"""Z3 encoding for SAD-05 — auto-extracted from vyapti_benchmark_z3_encodings.md"""

PROBLEM_ID = "SAD-05"


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

# f(n) = f(n-1) + f(n) for n > 1
# At n=2: f(2) = f(1) + f(2) = 1 + f(2)
# This means f(2) - f(2) = 1, i.e., 0 = 1 — CONTRADICTION

f2 = Int('f2')

s = Solver()
s.add(f2 == 1 + f2)  # f(2) = f(1) + f(2) = 1 + f(2)

print(f"SAD-05: System satisfiable? {s.check() == sat}")  # UNSAT — contradiction
'''


if __name__ == "__main__":
    result = check()
    print(result)
