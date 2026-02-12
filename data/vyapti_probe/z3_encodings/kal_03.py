"""Z3 encoding for KAL-03 — auto-extracted from vyapti_benchmark_z3_encodings.md"""

PROBLEM_ID = "KAL-03"


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

# Fall=5 books, Summer=2 books. Currently summer. Student has 2.
# Q: Can check out 3rd? A: No

season = Int('season')  # 0=summer, 1=fall
limit = Int('limit')
current_count = Int('current_count')
can_checkout = Bool('can_checkout')

s = Solver()

s.add(season == 0)  # Summer
s.add(limit == 2)
s.add(current_count == 2)

# Can check out 3rd? No — at limit
s.add(can_checkout == (current_count < limit))

s.check()
m = s.model()
print(f"KAL-03: Can check out 3rd? {m[can_checkout]}")  # False
'''


if __name__ == "__main__":
    result = check()
    print(result)
