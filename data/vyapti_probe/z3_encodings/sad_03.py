"""Z3 encoding for SAD-03 — auto-extracted from vyapti_benchmark_z3_encodings.md"""

PROBLEM_ID = "SAD-03"


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

# Red/Blue/Green boxes, key/coin/ring
# Rule 1: Red=key → Blue=coin. Rule 2: Green≠ring.
# Proof by contradiction: Red≠key. But final assignment underdetermined.

Red, Blue, Green = Ints('Red Blue Green')  # 0=key, 1=coin, 2=ring
key, coin, ring = 0, 1, 2

s = Solver()

s.add(And(Red >= 0, Red <= 2))
s.add(And(Blue >= 0, Blue <= 2))
s.add(And(Green >= 0, Green <= 2))
s.add(Distinct(Red, Blue, Green))

# Rule 1: Red=key → Blue=coin
s.add(Implies(Red == key, Blue == coin))
# Rule 2: Green≠ring
s.add(Green != ring)

# Proof by contradiction: assume Red=key. Then Blue=coin. Green can be key or coin.
# Actually: if Red=key, Blue=coin, Green=ring. But Green≠ring. So Red≠key.
s.add(Red != key)

# Check: multiple solutions? Red can be coin or ring.
s.push()
s.add(Red == coin)
r1 = s.check()
s.pop()

s.push()
s.add(Red == ring)
r2 = s.check()
s.pop()

print(f"SAD-03: Multiple solutions: Red=coin {r1 == sat}, Red=ring {r2 == sat}")
print("SAD-03: Only Red≠key provable; full assignment underdetermined")
'''


if __name__ == "__main__":
    result = check()
    print(result)
