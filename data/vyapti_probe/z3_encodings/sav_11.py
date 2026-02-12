"""Z3 encoding for SAV-11 — auto-extracted from vyapti_benchmark_z3_encodings.md"""

PROBLEM_ID = "SAV-11"


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

Book = DeclareSort('Book')
alg, bio, comp, db, enc, frac = Consts('alg bio comp db enc frac', Book)

recommended = Function('recommended', Book, BoolSort())
available = Function('available', Book, BoolSort())

s = Solver()

s.add(recommended(alg), available(alg))
s.add(recommended(bio), available(bio))
s.add(recommended(comp), available(comp))
s.add(recommended(db), available(db))
s.add(recommended(enc), Not(available(enc)))  # KEY: Encryption recommended, NOT available
s.add(Not(recommended(frac)), available(frac))

# Vyāpti: recommended → available
s.push()
s.add(recommended(enc), Not(available(enc)))
print(f"SAV-11: 'Recommended ⊆ Available' holds: {s.check() != sat}")  # False
s.pop()

# Correct answer: No (Encryption)
'''


if __name__ == "__main__":
    result = check()
    print(result)
