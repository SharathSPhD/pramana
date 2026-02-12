"""Z3 encoding for SAV-04-C — auto-extracted from vyapti_benchmark_z3_encodings.md"""

PROBLEM_ID = "SAV-04-C"


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

Student = DeclareSort('Student')
Mira, Neel, Omar, Priya, Qasim, Riya, Sana, Tara = Consts(
    'Mira Neel Omar Priya Qasim Riya Sana Tara', Student)

P = Function('Philosophy', Student, BoolSort())
M = Function('Mathematics', Student, BoolSort())
L = Function('Logic', Student, BoolSort())

s = Solver()

# P={Mira,Neel,Omar,Priya,Riya}, M={Mira,Neel,Omar,Priya,Sana}, L={Mira,Neel,Omar,Priya,Qasim,Tara}
for st in [Mira, Neel, Omar, Priya, Riya]: s.add(P(st))
for st in [Mira, Neel, Omar, Priya, Sana]: s.add(M(st))
for st in [Mira, Neel, Omar, Priya, Qasim, Tara]: s.add(L(st))

s.add(Not(P(Qasim)), Not(P(Sana)), Not(P(Tara)))
s.add(Not(M(Riya)), Not(M(Qasim)), Not(M(Tara)))
s.add(Not(L(Riya)), Not(L(Sana)))

# P∩M = {Mira,Neel,Omar,Priya} ⊆ L — all in L
s.push()
for st in [Mira, Neel, Omar, Priya]:
    s.add(P(st), M(st), Not(L(st)))
print(f"SAV-04-C: P∩M ⊆ L violated? {s.check() == sat}")  # UNSAT — subset holds
s.pop()

# Correct answer: Yes
'''


if __name__ == "__main__":
    result = check()
    print(result)
