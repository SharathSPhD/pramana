"""Z3 encoding for SAV-03-C — auto-extracted from vyapti_benchmark_z3_encodings.md"""

PROBLEM_ID = "SAV-03-C"


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

S1 = [Bool(f'S1_t{i}') for i in range(1, 6)]
Light = [Bool(f'Light_t{i}') for i in range(1, 6)]

s = Solver()

# All tests: S1=ON → Light=ON
s.add(S1[0], Light[0])
s.add(S1[1], Light[1])
s.add(S1[2], Light[2])
s.add(S1[3], Light[3])  # CHANGED: Test 4 now Light=ON
s.add(Not(S1[4]), Not(Light[4]))

# Vyāpti holds for all tests
vyapti_holds = And(*[Implies(S1[i], Light[i]) for i in range(5)])
s.push()
s.add(vyapti_holds)
print(f"SAV-03-C: Vyāpti 'S1→Light' holds: {s.check() == sat}")  # SAT
s.pop()

# Correct answer: Yes
'''


if __name__ == "__main__":
    result = check()
    print(result)
