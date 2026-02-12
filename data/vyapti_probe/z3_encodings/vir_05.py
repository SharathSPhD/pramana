"""Z3 encoding for VIR-05 — auto-extracted from vyapti_benchmark_z3_encodings.md"""

PROBLEM_ID = "VIR-05"


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

# A={p,q,r,s,t}, B={q,r,u,v}
# Policy: sports ∧ ¬honor_roll → study_hall
# sports = A, honor_roll = B
# Who must attend? A ∩ B' = {p,s,t}

p, q, r, s, t, u, v = Bools('p q r s t u v')

in_A = Function('in_A', IntSort(), BoolSort())  # Use indices
# Simpler: A members p,q,r,s,t; B members q,r,u,v
# A∩B' = who is in A but not B = p, s, t

# Encode: sport(x) means x in A, honor_roll(x) means x in B
Person = DeclareSort('Person')
pa, pq, pr, ps, pt, pu, pv = Consts('pa pq pr ps pt pu pv', Person)

sports = Function('sports', Person, BoolSort())
honor_roll = Function('honor_roll', Person, BoolSort())
study_hall = Function('study_hall', Person, BoolSort())

s = Solver()

# A = {p,q,r,s,t}, B = {q,r,u,v}
for x in [pa, pq, pr, ps, pt]: s.add(sports(x))
for x in [pq, pr, pu, pv]: s.add(honor_roll(x))
s.add(Not(sports(pu)), Not(sports(pv)))
s.add(Not(honor_roll(pa)), Not(honor_roll(ps)), Not(honor_roll(pt)))

# Policy: sports ∧ ¬honor_roll → study_hall
x = Const('x', Person)
s.add(ForAll([x], Implies(And(sports(x), Not(honor_roll(x))), study_hall(x))))

# Who must attend? pa, ps, pt (in A, not in B)
s.push()
s.add(Not(study_hall(pa)))
print(f"VIR-05: Can pa avoid study_hall? {s.check() == sat}")  # UNSAT
s.pop()

# Correct answer: A∩B' = {p,s,t}
print("VIR-05: Must attend: pa, ps, pt (A∩B')")
'''


if __name__ == "__main__":
    result = check()
    print(result)
