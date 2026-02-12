"""Z3 encoding for SAV-06 — auto-extracted from vyapti_benchmark_z3_encodings.md"""

PROBLEM_ID = "SAV-06"


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
s.add(labrador(Duke), Not(friendly(Duke)))  # KEY: Duke is Lab, NOT friendly
s.add(Not(labrador(Coco)), friendly(Coco))

# Check vyāpti: labrador → friendly
print("=== SAV-06: Checking 'All Labs friendly' ===")
s.push()
s.add(labrador(Duke), Not(friendly(Duke)))
print(f"Counterexample exists (Duke): {s.check() == sat}")  # SAT
s.pop()

# Correct answer: No (Duke)
# Trap answer: Yes
'''


if __name__ == "__main__":
    result = check()
    print(result)
