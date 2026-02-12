"""Z3 encoding for VIR-09-C — auto-extracted from vyapti_benchmark_z3_encodings.md"""

PROBLEM_ID = "VIR-09-C"


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

high_priority = Bool('high_priority')
senior_partner = Bool('senior_partner')

s = Solver()
s.add(Implies(high_priority, senior_partner))
s.add(Not(senior_partner))  # Fact: not senior partner

# Modus tollens: ¬senior_partner → ¬high_priority
s.push()
s.add(high_priority)
print(f"VIR-09-C: Can case be high priority? {s.check() == sat}")  # UNSAT
s.pop()

# Correct answer: Yes (¬high_priority)
'''


if __name__ == "__main__":
    result = check()
    print(result)
