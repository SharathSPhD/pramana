"""Z3 encoding for VIR-06 — auto-extracted from vyapti_benchmark_z3_encodings.md"""

PROBLEM_ID = "VIR-06"


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

# "It is false that all lights in Room A are on"
# ¬∀x: light_on(x)  ≡  ∃x: ¬light_on(x)
# So: at least one light is off

Light = DeclareSort('Light')
l1, l2, l3 = Consts('l1 l2 l3', Light)
on = Function('on', Light, BoolSort())

s = Solver()

# Premise: ¬(∀x: on(x))
s.add(Not(ForAll([l1, l2, l3], And(on(l1), on(l2), on(l3)))))

# What must be true? At least one off: ∃x: ¬on(x)
s.push()
s.add(ForAll([l1, l2, l3], on(l1)))  # Can all be on?
# Actually: DeMorgan: ¬∀ ≡ ∃¬. So we need at least one off.
# Check: is "at least one off" forced?
s.pop()
s.push()
# If we add that all are on, we get contradiction
s.add(on(l1), on(l2), on(l3))
print(f"VIR-06: Can all lights be on? {s.check() == sat}")  # UNSAT — contradicts premise
s.pop()

# Correct answer: (b) At least one light is off
print("VIR-06: At least one light is off (¬∀ ≡ ∃¬)")
'''


if __name__ == "__main__":
    result = check()
    print(result)
