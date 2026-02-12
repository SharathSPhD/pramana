"""Z3 encoding for SAV-09 — auto-extracted from vyapti_benchmark_z3_encodings.md"""

PROBLEM_ID = "SAV-09"


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

# Events 1-5: A_triggered, alarm state
A_trig = [Bool(f'A_t{i}') for i in range(1, 6)]
alarm = [Bool(f'alarm_t{i}') for i in range(1, 6)]

s = Solver()

s.add(A_trig[0], alarm[0])   # Event 1
s.add(A_trig[1], alarm[1])   # Event 2
s.add(A_trig[2], alarm[2])   # Event 3
s.add(A_trig[3], Not(alarm[3]))  # Event 4: A triggered alone, alarm OFF
s.add(Not(A_trig[4]), alarm[4])  # Event 5

# Vyāpti: A triggered → alarm
vyapti = And(*[Implies(A_trig[i], alarm[i]) for i in range(5)])
s.push()
s.add(vyapti)
print(f"SAV-09: Vyāpti 'A→alarm' holds: {s.check() == sat}")  # False — Event 4
s.pop()

# Correct answer: No (Event 4)
'''


if __name__ == "__main__":
    result = check()
    print(result)
