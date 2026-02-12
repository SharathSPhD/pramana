"""Z3 encoding for SAV-04 — auto-extracted from vyapti_benchmark_z3_encodings.md"""

PROBLEM_ID = "SAV-04"


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

# Students as enumeration
Student = DeclareSort('Student')
Mira, Neel, Omar, Priya, Qasim, Riya, Sana, Tara = Consts(
    'Mira Neel Omar Priya Qasim Riya Sana Tara', Student)

P = Function('Philosophy', Student, BoolSort())
M = Function('Mathematics', Student, BoolSort())
L = Function('Logic', Student, BoolSort())

s = Solver()

# Set memberships
for st in [Mira, Neel, Omar, Priya, Qasim, Riya]: s.add(P(st))
for st in [Mira, Neel, Omar, Qasim, Sana]: s.add(M(st))
for st in [Mira, Neel, Omar, Priya, Tara]: s.add(L(st))

# Non-memberships
s.add(Not(P(Sana)), Not(P(Tara)))
s.add(Not(M(Priya)), Not(M(Riya)), Not(M(Tara)))
s.add(Not(L(Qasim)), Not(L(Riya)), Not(L(Sana)))  # KEY: Qasim NOT in Logic

# Check vyāpti: P(x) ∧ M(x) → L(x)
# Qasim is in P ∩ M but NOT in L
s.push()
s.add(P(Qasim), M(Qasim), Not(L(Qasim)))
print(f"SAV-04: Counterexample exists (Qasim): {s.check() == sat}")  # SAT — P∩M ⊆ L fails
s.pop()

# Correct answer: No (Qasim ∈ P∩M but ∉ L)
'''


if __name__ == "__main__":
    result = check()
    print(result)
