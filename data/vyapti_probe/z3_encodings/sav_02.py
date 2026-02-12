"""Z3 encoding for SAV-02 — auto-extracted from vyapti_benchmark_z3_encodings.md"""

PROBLEM_ID = "SAV-02"


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
s.add(nocturnal(D), Not(echolocation(D)))  # KEY counterexample
s.add(Not(nocturnal(E)), Not(echolocation(E)))
s.add(nocturnal(F), echolocation(F))
s.add(nocturnal(G), echolocation(G))
s.add(nocturnal(H))  # H is nocturnal, echolocation unknown

# Check vyāpti: nocturnal → echolocation
print("=== SAV-02: Checking vyāpti 'Nocturnal → Echolocation' ===")
# Species D is nocturnal without echolocation
s.push()
s.add(Not(echolocation(H)))
print(f"H can lack echolocation: {s.check() == sat}")  # SAT — vyāpti doesn't force it
s.pop()

# Correct answer: No (D is counterexample)
# Trap answer: Yes (vyāpti wrongly assumed)
'''


if __name__ == "__main__":
    result = check()
    print(result)
