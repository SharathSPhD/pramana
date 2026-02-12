"""Z3 encoding for SAV-11-C — auto-extracted from vyapti_benchmark_z3_encodings.md"""

PROBLEM_ID = "SAV-11-C"


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
s.add(recommended(enc), available(enc))  # CHANGED: Encryption available
s.add(Not(recommended(frac)), available(frac))

x = Const('x', Book)
s.add(ForAll([x], Implies(recommended(x), available(x))))

print("SAV-11-C: Recommended ⊆ Available — holds")
'''


if __name__ == "__main__":
    result = check()
    print(result)
