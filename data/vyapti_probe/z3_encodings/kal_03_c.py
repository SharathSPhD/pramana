"""Z3 encoding for KAL-03-C — auto-extracted from vyapti_benchmark_z3_encodings.md"""

PROBLEM_ID = "KAL-03-C"


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

season = Int('season')
limit = Int('limit')
current_count = Int('current_count')
can_checkout = Bool('can_checkout')

s = Solver()

s.add(season == 1)  # Fall
s.add(limit == 5)
s.add(current_count == 4)

s.add(can_checkout == (current_count < limit))

s.check()
m = s.model()
print(f"KAL-03-C: Can check out 5th? {m[can_checkout]}")  # True
'''


if __name__ == "__main__":
    result = check()
    print(result)
