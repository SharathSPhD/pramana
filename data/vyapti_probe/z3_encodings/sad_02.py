"""Z3 encoding for SAD-02 — auto-extracted from vyapti_benchmark_z3_encodings.md"""

PROBLEM_ID = "SAD-02"


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
s.add(X == Y)  # X ↔ Y
s.add(Y == X)  # Y ↔ X (redundant)

# Check: is X determined?
s.push()
s.add(X == True)
r1 = s.check()
s.pop()

s.push()
s.add(X == False)
r2 = s.check()
s.pop()

print(f"SAD-02: X can be True: {r1 == sat}, X can be False: {r2 == sat}")
# Both SAT — X is underdetermined (two solutions: both True or both False)
'''


if __name__ == "__main__":
    result = check()
    print(result)
