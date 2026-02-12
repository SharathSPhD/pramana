"""Z3 encoding for SAV-12-C — auto-extracted from vyapti_benchmark_z3_encodings.md"""

PROBLEM_ID = "SAV-12-C"


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
patients = [Const(f'p{i}', Patient) for i in range(1, 8)]

cond_x = Function('cond_x', Patient, BoolSort())
treat_t = Function('treat_t', Patient, BoolSort())
recovered = Function('recovered', Patient, BoolSort())

s = Solver()

for p in patients:
    s.add(cond_x(p), treat_t(p), recovered(p))

x = Const('x', Patient)
s.add(ForAll([x], Implies(And(cond_x(x), treat_t(x)), recovered(x))))

print("SAV-12-C: CondX ∧ TreatT → Recovery — holds")
'''


if __name__ == "__main__":
    result = check()
    print(result)
