"""Z3 encoding for SAV-06-C — auto-extracted from vyapti_benchmark_z3_encodings.md"""

PROBLEM_ID = "SAV-06-C"


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

Dog = DeclareSort('Dog')
Max, Bella, Rocky, Luna, Buddy, Duke, Coco = Consts(
    'Max Bella Rocky Luna Buddy Duke Coco', Dog)

labrador = Function('labrador', Dog, BoolSort())
friendly = Function('friendly', Dog, BoolSort())

s = Solver()

s.add(labrador(Max), friendly(Max))
s.add(labrador(Bella), friendly(Bella))
s.add(labrador(Rocky), friendly(Rocky))
s.add(labrador(Luna), friendly(Luna))
s.add(labrador(Buddy), friendly(Buddy))
s.add(labrador(Duke), friendly(Duke))  # CHANGED: Duke is friendly
s.add(Not(labrador(Coco)), friendly(Coco))

# Explicit rule: all Labradors friendly
x = Const('x', Dog)
s.add(ForAll([x], Implies(labrador(x), friendly(x))))

# All Labs forced to be friendly
s.push()
s.add(labrador(Duke), Not(friendly(Duke)))
print(f"SAV-06-C: Duke can be unfriendly? {s.check() == sat}")  # UNSAT
s.pop()

# Correct answer: Yes
'''


if __name__ == "__main__":
    result = check()
    print(result)
