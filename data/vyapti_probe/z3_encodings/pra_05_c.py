"""Z3 encoding for PRA-05-C — auto-extracted from vyapti_benchmark_z3_encodings.md"""

PROBLEM_ID = "PRA-05-C"


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

Priya, Qadir, Ravi, Sita = Ints('Priya Qadir Ravi Sita')

s = Solver()

for p in [Priya, Qadir, Ravi, Sita]:
    s.add(Or(p == 0, p == 1))  # 0=North, 1=South

s.add(Priya == 0)
s.add(Qadir == 1)
s.add(Sita == 0)
# South has exactly 2. Qadir=South(1). With Priya=0,Sita=0, need Ravi=1.
s.add(Priya + Qadir + Ravi + Sita == 2)

s.check()
m = s.model()
print(f"PRA-05-C: Ravi = South: {m[Ravi] == 1}")
'''


if __name__ == "__main__":
    result = check()
    print(result)
