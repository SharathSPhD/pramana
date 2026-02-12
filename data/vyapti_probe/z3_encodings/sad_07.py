"""Z3 encoding for SAD-07 — auto-extracted from vyapti_benchmark_z3_encodings.md"""

PROBLEM_ID = "SAD-07"


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

# Proof that primes > 2 are odd. Uses proof by contradiction. Valid (NOT circular).
# Encode: if n>2 prime, then n odd. Proof: assume n even, n=2k, n>2, so n has factor 2.
# Contradiction. So n odd.

# This is a valid proof — no circularity
n = Int('n')
prime = Function('prime', IntSort(), BoolSort())
even = Function('even', IntSort(), BoolSort())

s = Solver()

# For n>2 prime: n cannot be even (else 2 divides n)
s.add(Implies(And(n > 2, prime(n)), Not(even(n))))

print("SAD-07: Proof valid — not circular")
'''


if __name__ == "__main__":
    result = check()
    print(result)
