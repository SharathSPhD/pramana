"""Z3 encoding for SAV-02-C — auto-extracted from vyapti_benchmark_z3_encodings.md"""

PROBLEM_ID = "SAV-02-C"


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

Species = DeclareSort('Species')
A, B, C, D, E, F, G, H = Consts('A B C D E F G H', Species)

nocturnal = Function('nocturnal', Species, BoolSort())
echolocation = Function('echolocation', Species, BoolSort())

s = Solver()

s.add(nocturnal(A), echolocation(A))
s.add(nocturnal(B), echolocation(B))
s.add(nocturnal(C), echolocation(C))
s.add(nocturnal(D), echolocation(D))  # CHANGED: D now has echolocation
s.add(Not(nocturnal(E)), Not(echolocation(E)))
s.add(nocturnal(F), echolocation(F))
s.add(nocturnal(G), echolocation(G))
s.add(nocturnal(H))

# Explicit rule: all nocturnal species have echolocation
x = Const('x', Species)
s.add(ForAll([x], Implies(nocturnal(x), echolocation(x))))

# H is forced to have echolocation
s.push()
s.add(Not(echolocation(H)))
print(f"SAV-02-C: H can lack echolocation: {s.check() == sat}")  # UNSAT — H must have echolocation
s.pop()

# Correct answer: Yes
'''


if __name__ == "__main__":
    result = check()
    print(result)
