"""Z3 encoding for SAV-01-C — auto-extracted from vyapti_benchmark_z3_encodings.md"""

PROBLEM_ID = "SAV-01-C"


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

Employee = DeclareSort('Employee')
Aarav, Bina, Chetan, Diya, Esha, Farid, Gita = Consts(
    'Aarav Bina Chetan Diya Esha Farid Gita', Employee)

engineering = Function('engineering', Employee, BoolSort())
remote = Function('remote', Employee, BoolSort())

s = Solver()
s.add(Distinct(*[Aarav, Bina, Chetan, Diya, Esha, Farid, Gita]))

# Premises (Chetan now remote) + explicit rule
s.add(engineering(Aarav), remote(Aarav))
s.add(engineering(Bina), remote(Bina))
s.add(engineering(Chetan), remote(Chetan))  # CHANGED: Chetan is remote
s.add(Not(engineering(Diya)))
s.add(engineering(Esha), remote(Esha))
s.add(engineering(Farid), remote(Farid))
s.add(engineering(Gita))

# Explicit rule: all engineering employees work remotely
x = Const('x', Employee)
s.add(ForAll([x], Implies(engineering(x), remote(x))))

# Gita is forced to be remote
s.push()
s.add(Not(remote(Gita)))
result = s.check()
print(f"SAV-01-C: Gita can be non-remote: {result == sat}")  # Should be UNSAT
s.pop()
'''


if __name__ == "__main__":
    result = check()
    print(result)
