"""Z3 encoding for PRA-08 — auto-extracted from vyapti_benchmark_z3_encodings.md"""

PROBLEM_ID = "PRA-08"


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

# Window∨Door. Window→mud. ¬mud. Door→scratches. Scratches found.
# Dog is loud, didn't bark — irrelevant.
# Q: How? A: Door

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

# Dog clues — not encoded (irrelevant)

s.push()
s.add(door)
print(f"PRA-08: Door consistent? {s.check() == sat}")  # SAT
s.pop()

s.push()
s.add(window)
print(f"PRA-08: Window consistent? {s.check() == sat}")  # UNSAT (mud false)
s.pop()

print("PRA-08: Answer = Door")
'''


if __name__ == "__main__":
    result = check()
    print(result)
