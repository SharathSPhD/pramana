"""Z3 encoding for PRA-09 — auto-extracted from vyapti_benchmark_z3_encodings.md"""

PROBLEM_ID = "PRA-09"


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

# 6 subjects: handedness + chocolate preference. No rule/correlation.
# Q: Does handedness predict chocolate? A: No (no vyāpti)

Subject = DeclareSort('Subject')
s1, s2, s3, s4, s5, s6 = Consts('s1 s2 s3 s4 s5 s6', Subject)

left_handed = Function('left_handed', Subject, BoolSort())
prefers_dark = Function('prefers_dark', Subject, BoolSort())

s = Solver()

# Some correlation but no universal rule — we can have left without dark, etc.
# Check: does left_handed(x) → prefers_dark(x) hold?
# With mixed data, vyāpti fails.
x = Const('x', Subject)
s.add(Not(ForAll([x], Implies(left_handed(x), prefers_dark(x)))))

print("PRA-09: No vyāpti — handedness does not predict chocolate")
'''


if __name__ == "__main__":
    result = check()
    print(result)
