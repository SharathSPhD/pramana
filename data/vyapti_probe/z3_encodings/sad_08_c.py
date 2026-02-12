"""Z3 encoding for SAD-08-C — auto-extracted from vyapti_benchmark_z3_encodings.md"""

PROBLEM_ID = "SAD-08-C"


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

T1, T2, T3 = Ints('T1 T2 T3')

s = Solver()
for t in [T1, T2, T3]:
    s.add(And(t >= 1, t <= 3))
s.add(Distinct(T1, T2, T3))

s.add(T2 == T1 + 1)  # T1 before T2
s.add(T3 == T2 + 1)  # T2 before T3

s.check()
m = s.model()
print(f"SAD-08-C: T1={m[T1]}, T2={m[T2]}, T3={m[T3]}")  # 1, 2, 3
'''


if __name__ == "__main__":
    result = check()
    print(result)
