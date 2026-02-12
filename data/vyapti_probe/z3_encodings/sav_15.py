"""Z3 encoding for SAV-15 — auto-extracted from vyapti_benchmark_z3_encodings.md"""

PROBLEM_ID = "SAV-15"


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

Week = DeclareSort('Week')
w1, w2, w3, w4, w5, w6 = Consts('w1 w2 w3 w4 w5 w6', Week)

M = Function('M', Week, BoolSort())
on_time = Function('on_time', Week, BoolSort())

s = Solver()

for i, w in enumerate([w1, w2, w3, w4]):
    s.add(M(w), on_time(w))
s.add(M(w5), Not(on_time(w5)))  # Week 5: M=true, NOT on time
s.add(Not(M(w6)), on_time(w6))

# Vyāpti: M → on_time
s.push()
s.add(M(w5), Not(on_time(w5)))
print(f"SAV-15: 'M → on time' holds: {s.check() != sat}")  # False
s.pop()

# Correct answer: No (Week 5)
'''


if __name__ == "__main__":
    result = check()
    print(result)
