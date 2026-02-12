"""Z3 encoding for SAD-02-C — auto-extracted from vyapti_benchmark_z3_encodings.md"""

PROBLEM_ID = "SAD-02-C"


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

X, Y = Bools('X Y')

s = Solver()
s.add(X == Y)
s.add(Y == True)  # Independent anchor

s.push()
s.add(X == False)
print(f"SAD-02-C: X can be False? {s.check() == sat}")  # UNSAT — X must be True
s.pop()
'''


if __name__ == "__main__":
    result = check()
    print(result)
