"""Z3 encoding for KAL-04 — auto-extracted from vyapti_benchmark_z3_encodings.md"""

PROBLEM_ID = "KAL-04"


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

# Time-indexed access
query_date = Int('query_date')  # Day of year
grant_date = Int('grant_date')
revoke_date = Int('revoke_date')
has_access = Bool('has_access')

s = Solver()

s.add(grant_date == 1)    # Jan 1
s.add(revoke_date == 74)  # Mar 15
s.add(query_date == 152)  # Jun 1

# Access rule: access if grant_date ≤ query_date < revoke_date
s.add(has_access == And(query_date >= grant_date, query_date < revoke_date))

s.check()
m = s.model()
print(f"KAL-04: Has access on Jun 1? {m[has_access]}")  # False
'''


if __name__ == "__main__":
    result = check()
    print(result)
