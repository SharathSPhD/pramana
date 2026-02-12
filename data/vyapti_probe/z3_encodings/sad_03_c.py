"""Z3 encoding for SAD-03-C — auto-extracted from vyapti_benchmark_z3_encodings.md"""

PROBLEM_ID = "SAD-03-C"


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

Red, Blue, Green = Ints('Red Blue Green')
key, coin, ring = 0, 1, 2

s = Solver()

s.add(And(Red >= 0, Red <= 2))
s.add(And(Blue >= 0, Blue <= 2))
s.add(And(Green >= 0, Green <= 2))
s.add(Distinct(Red, Blue, Green))

s.add(Implies(Red == key, Blue == coin))
s.add(Green == key)
s.add(Red == ring)

s.check()
m = s.model()
print(f"SAD-03-C: Green=key, Red=ring, Blue=coin")
print(f"  Green={m[Green]}, Red={m[Red]}, Blue={m[Blue]}")
'''


if __name__ == "__main__":
    result = check()
    print(result)
