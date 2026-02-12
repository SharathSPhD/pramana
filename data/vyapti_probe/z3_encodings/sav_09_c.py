"""Z3 encoding for SAV-09-C — auto-extracted from vyapti_benchmark_z3_encodings.md"""

PROBLEM_ID = "SAV-09-C"


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

A_trig = [Bool(f'A_t{i}') for i in range(1, 6)]
alarm = [Bool(f'alarm_t{i}') for i in range(1, 6)]

s = Solver()

s.add(A_trig[0], alarm[0])
s.add(A_trig[1], alarm[1])
s.add(A_trig[2], alarm[2])
s.add(A_trig[3], alarm[3])  # CHANGED: Event 4 alarm ON
s.add(Not(A_trig[4]), alarm[4])

vyapti = And(*[Implies(A_trig[i], alarm[i]) for i in range(5)])
s.push()
s.add(vyapti)
print(f"SAV-09-C: Vyāpti 'A→alarm' holds: {s.check() == sat}")  # SAT
s.pop()

# Correct answer: Yes
'''


if __name__ == "__main__":
    result = check()
    print(result)
