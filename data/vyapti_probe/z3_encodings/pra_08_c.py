"""Z3 encoding for PRA-08-C — auto-extracted from vyapti_benchmark_z3_encodings.md"""

PROBLEM_ID = "PRA-08-C"


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

window = Bool('window')
door = Bool('door')
mud = Bool('mud')
scratches = Bool('scratches')

s = Solver()

s.add(Or(window, door))
s.add(Implies(window, mud))
s.add(Not(mud))
s.add(Implies(door, scratches))
s.add(scratches)

s.push()
s.add(Not(door))
print(f"PRA-08-C: Can it NOT be door? {s.check() == sat}")  # UNSAT
s.pop()

print("PRA-08-C: Answer = Door")
'''


if __name__ == "__main__":
    result = check()
    print(result)
