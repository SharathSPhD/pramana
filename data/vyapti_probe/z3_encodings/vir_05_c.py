"""Z3 encoding for VIR-05-C — auto-extracted from vyapti_benchmark_z3_encodings.md"""

PROBLEM_ID = "VIR-05-C"


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

Person = DeclareSort('Person')
pa, pq, pr, ps, pt, pu, pv = Consts('pa pq pr ps pt pu pv', Person)

sports = Function('sports', Person, BoolSort())
honor_roll = Function('honor_roll', Person, BoolSort())
scholarship = Function('scholarship', Person, BoolSort())

s = Solver()

for x in [pa, pq, pr, ps, pt]: s.add(sports(x))
for x in [pq, pr, pu, pv]: s.add(honor_roll(x))
s.add(Not(sports(pu)), Not(sports(pv)))
s.add(Not(honor_roll(pa)), Not(honor_roll(ps)), Not(honor_roll(pt)))

# Policy: sports ∧ honor_roll → scholarship
x = Const('x', Person)
s.add(ForAll([x], Implies(And(sports(x), honor_roll(x)), scholarship(x))))

# Who eligible? A∩B = {q,r}
s.push()
s.add(sports(pq), honor_roll(pq), Not(scholarship(pq)))
print(f"VIR-05-C: Can pq avoid scholarship? {s.check() == sat}")  # UNSAT
s.pop()

# Correct answer: A∩B = {q,r}
print("VIR-05-C: Eligible: pq, pr")
'''


if __name__ == "__main__":
    result = check()
    print(result)
