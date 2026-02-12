"""Z3 encoding for SAV-07 — auto-extracted from vyapti_benchmark_z3_encodings.md"""

PROBLEM_ID = "SAV-07"


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
s.add(contains_X(g), Not(acidic(g)), Not(reacts(g)))  # KEY: contains X but basic, no reaction
s.add(contains_X(d), acidic(d), reacts(d))
s.add(contains_X(e), acidic(e), reacts(e))
s.add(Not(contains_X(z)), acidic(z), Not(reacts(z)))

# Check vyāpti: contains_X → reacts
print("=== SAV-07: Checking 'contains X → reacts' ===")
s.push()
s.add(contains_X(g), Not(reacts(g)))  # Already asserted
print(f"Counterexample exists (gamma): {s.check() == sat}")  # SAT
s.pop()

# The ACTUAL pattern is: contains_X ∧ acidic → reacts
# Check this refined vyāpti
print("Checking refined 'contains X ∧ acidic → reacts':")
all_acidic_X_react = And(
    Implies(And(contains_X(a), acidic(a)), reacts(a)),
    Implies(And(contains_X(b), acidic(b)), reacts(b)),
    Implies(And(contains_X(g), acidic(g)), reacts(g)),  # gamma is NOT acidic, so vacuously true
    Implies(And(contains_X(d), acidic(d)), reacts(d)),
    Implies(And(contains_X(e), acidic(e)), reacts(e)),
)
s.push()
s.add(all_acidic_X_react)
print(f"Refined vyāpti consistent: {s.check() == sat}")  # SAT
s.pop()

# Correct answer: No (γ counterexample)
'''


if __name__ == "__main__":
    result = check()
    print(result)
