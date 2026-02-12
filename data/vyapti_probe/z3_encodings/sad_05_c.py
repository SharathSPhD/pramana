"""Z3 encoding for SAD-05-C — auto-extracted from vyapti_benchmark_z3_encodings.md"""

PROBLEM_ID = "SAD-05-C"


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

f1, f2, f3, f4 = Ints('f1 f2 f3 f4')

s = Solver()
s.add(f1 == 1)
s.add(f2 == f1 + 2)  # f(n) = f(n-1) + n
s.add(f3 == f2 + 3)
s.add(f4 == f3 + 4)

s.check()
m = s.model()
print(f"SAD-05-C: f(4) = {m[f4]}")  # Should be 10
'''


if __name__ == "__main__":
    result = check()
    print(result)
