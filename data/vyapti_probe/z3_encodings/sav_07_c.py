"""Z3 encoding for SAV-07-C — auto-extracted from vyapti_benchmark_z3_encodings.md"""

PROBLEM_ID = "SAV-07-C"


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

Compound = DeclareSort('Compound')
a, b, g, d, e, z = Consts('alpha beta gamma delta epsilon zeta', Compound)

contains_X = Function('contains_X', Compound, BoolSort())
acidic = Function('acidic', Compound, BoolSort())
reacts = Function('reacts', Compound, BoolSort())

s = Solver()

s.add(contains_X(a), acidic(a), reacts(a))
s.add(contains_X(b), acidic(b), reacts(b))
s.add(contains_X(g), acidic(g), reacts(g))  # CHANGED: γ now acidic and reactive
s.add(contains_X(d), acidic(d), reacts(d))
s.add(contains_X(e), acidic(e), reacts(e))
s.add(Not(contains_X(z)), acidic(z), Not(reacts(z)))

# Vyāpti: contains_X → reacts (all X-containing compounds now react)
x = Const('x', Compound)
s.add(ForAll([x], Implies(contains_X(x), reacts(x))))

s.push()
s.add(contains_X(g), Not(reacts(g)))
print(f"SAV-07-C: γ can fail to react? {s.check() == sat}")  # UNSAT
s.pop()

# Correct answer: Yes
'''


if __name__ == "__main__":
    result = check()
    print(result)
