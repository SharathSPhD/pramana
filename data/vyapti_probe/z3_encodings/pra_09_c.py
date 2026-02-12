"""Z3 encoding for PRA-09-C — auto-extracted from vyapti_benchmark_z3_encodings.md"""

PROBLEM_ID = "PRA-09-C"


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

Subject = DeclareSort('Subject')
s1, s2, s3, s4, s5, s6 = Consts('s1 s2 s3 s4 s5 s6', Subject)

medication_x = Function('medication_x', Subject, BoolSort())
improved = Function('improved', Subject, BoolSort())

s = Solver()

# RCT: perfect correlation — medication_x → improved
x = Const('x', Subject)
s.add(ForAll([x], Implies(medication_x(x), improved(x))))

s.push()
s.add(medication_x(s1))
s.add(Not(improved(s1)))
print(f"PRA-09-C: Can medication fail to improve? {s.check() == sat}")  # UNSAT
s.pop()

print("PRA-09-C: Yes — medication predicts improvement")
'''


if __name__ == "__main__":
    result = check()
    print(result)
