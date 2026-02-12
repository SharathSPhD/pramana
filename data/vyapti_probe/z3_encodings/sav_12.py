"""Z3 encoding for SAV-12 — auto-extracted from vyapti_benchmark_z3_encodings.md"""

PROBLEM_ID = "SAV-12"


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

Patient = DeclareSort('Patient')
p1, p2, p3, p4, p5, p6, p7, p8 = Consts('p1 p2 p3 p4 p5 p6 p7 p8', Patient)

cond_x = Function('cond_x', Patient, BoolSort())
treat_t = Function('treat_t', Patient, BoolSort())
recovered = Function('recovered', Patient, BoolSort())

s = Solver()

# Patients 1-3, 5-7: CondX, TreatT, recovered
for p in [p1, p2, p3, p5, p6, p7]:
    s.add(cond_x(p), treat_t(p), recovered(p))
# Patient 4: CondX, TreatT, did NOT recover (counterexample)
s.add(cond_x(p4), treat_t(p4), Not(recovered(p4)))
# Patient 8: CondX, no treatment, recovered
s.add(cond_x(p8), Not(treat_t(p8)), recovered(p8))

# Vyāpti: cond_x ∧ treat_t → recovered
s.push()
s.add(cond_x(p4), treat_t(p4), Not(recovered(p4)))
print(f"SAV-12: 'CondX ∧ TreatT → Recovery' holds: {s.check() != sat}")  # False
s.pop()

# Correct answer: No (Patient 4)
'''


if __name__ == "__main__":
    result = check()
    print(result)
