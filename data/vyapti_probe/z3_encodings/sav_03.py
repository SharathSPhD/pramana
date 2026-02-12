"""Z3 encoding for SAV-03 — auto-extracted from vyapti_benchmark_z3_encodings.md"""

PROBLEM_ID = "SAV-03"


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

# 5 tests, each with S1 state and Light state
S1 = [Bool(f'S1_t{i}') for i in range(1, 6)]
Light = [Bool(f'Light_t{i}') for i in range(1, 6)]

s = Solver()

# Test results
s.add(S1[0], Light[0])       # Test 1: S1=ON, Light=ON
s.add(S1[1], Light[1])       # Test 2: S1=ON, Light=ON
s.add(S1[2], Light[2])       # Test 3: S1=ON, Light=ON
s.add(S1[3], Not(Light[3]))  # Test 4: S1=ON, Light=OFF ← counterexample
s.add(Not(S1[4]), Not(Light[4]))  # Test 5: S1=OFF, Light=OFF

# Check: does S1=ON → Light=ON hold for ALL tests?
vyapti_holds = And(*[Implies(S1[i], Light[i]) for i in range(5)])
s.push()
s.add(vyapti_holds)
print(f"SAV-03: Vyāpti 'S1→Light' holds: {s.check() == sat}")  # False — Test 4 contradicts (check=unsat)
s.pop()

# Correct answer: No (Test 4 counterexample)
'''


if __name__ == "__main__":
    result = check()
    print(result)
