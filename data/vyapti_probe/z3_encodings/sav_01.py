"""Z3 encoding for SAV-01 — auto-extracted from vyapti_benchmark_z3_encodings.md"""

PROBLEM_ID = "SAV-01"


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

# Entities
Employee = DeclareSort('Employee')
Aarav, Bina, Chetan, Diya, Esha, Farid, Gita = Consts(
    'Aarav Bina Chetan Diya Esha Farid Gita', Employee)

engineering = Function('engineering', Employee, BoolSort())
remote = Function('remote', Employee, BoolSort())

s = Solver()

# All distinct
employees = [Aarav, Bina, Chetan, Diya, Esha, Farid, Gita]
s.add(Distinct(*employees))

# Premises
s.add(engineering(Aarav), remote(Aarav))
s.add(engineering(Bina), remote(Bina))
s.add(engineering(Chetan), Not(remote(Chetan)))   # KEY: Chetan is on-site
s.add(Not(engineering(Diya)), remote(Diya))
s.add(engineering(Esha), remote(Esha))
s.add(engineering(Farid), remote(Farid))
s.add(engineering(Gita))  # Gita is in Engineering, remote status unknown

# Check vyāpti: "engineering(x) → remote(x)" for all x
# If we can find a model where engineering(x) ∧ ¬remote(x), vyāpti fails
print("=== SAV-01: Checking vyāpti 'Engineering → Remote' ===")
# Chetan is already a counterexample in the premises
s.push()
s.add(engineering(Chetan), Not(remote(Chetan)))  # Already asserted above
print(f"Vyāpti holds: {s.check() == unsat}")  # Should print False (SAT = counterexample exists)
s.pop()

# Check: is Gita forced to be remote?
s.push()
s.add(Not(remote(Gita)))
result = s.check()
print(f"Gita can be non-remote: {result == sat}")  # Should be SAT
s.pop()

# Correct answer: Gita is NOT forced to be remote
# Trap answer: Gita must be remote (WRONG)
'''


if __name__ == "__main__":
    result = check()
    print(result)
