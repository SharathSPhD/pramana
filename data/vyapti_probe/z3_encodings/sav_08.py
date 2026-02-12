"""Z3 encoding for SAV-08 — auto-extracted from vyapti_benchmark_z3_encodings.md"""

PROBLEM_ID = "SAV-08"


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

Student = DeclareSort('Student')
Arjun, Chirag, Durga, Fatima = Consts('Arjun Chirag Durga Fatima', Student)

physics = Function('physics', Student, BoolSort())
passed = Function('passed', Student, BoolSort())

s = Solver()

s.add(physics(Arjun), passed(Arjun))
s.add(physics(Chirag), passed(Chirag))
s.add(physics(Durga), Not(passed(Durga)))  # KEY: Durga (Physics) FAILED
s.add(physics(Fatima), passed(Fatima))

# Check vyāpti: physics → passed
print("=== SAV-08: Checking 'All Physics students passed' ===")
s.push()
s.add(physics(Durga), Not(passed(Durga)))
print(f"Counterexample exists (Durga): {s.check() == sat}")  # SAT
s.pop()

# Correct answer: No (Durga)
'''


if __name__ == "__main__":
    result = check()
    print(result)
