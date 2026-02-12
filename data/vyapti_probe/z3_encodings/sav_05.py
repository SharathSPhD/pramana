"""Z3 encoding for SAV-05 — auto-extracted from vyapti_benchmark_z3_encodings.md"""

PROBLEM_ID = "SAV-05"


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
Anita, Bhaskar, Chandra, Deepak, Ekta, Faizan = Consts(
    'Anita Bhaskar Chandra Deepak Ekta Faizan', Employee)

reports_to_vp = Function('reports_to_vp', Employee, BoolSort())
reports_to_coo = Function('reports_to_coo', Employee, BoolSort())
reports_to_ceo = Function('reports_to_ceo', Employee, BoolSort())

s = Solver()

# Rule: VP-reporters → CEO
e = Const('e', Employee)
s.add(ForAll([e], Implies(reports_to_vp(e), reports_to_ceo(e))))

# NO rule linking COO-reporters to CEO (this is the gap)

# Facts
for emp in [Anita, Bhaskar, Chandra, Deepak, Faizan]:
    s.add(reports_to_vp(emp))

s.add(Not(reports_to_vp(Ekta)))  # Ekta does NOT report to a VP
s.add(reports_to_coo(Ekta))      # Ekta reports to COO

# Check: is Ekta forced to report to CEO?
s.push()
s.add(Not(reports_to_ceo(Ekta)))
print(f"SAV-05: Ekta can NOT report to CEO: {s.check() == sat}")  # SAT — not forced
s.pop()

# Correct answer: No (Ekta's chain incomplete)
'''


if __name__ == "__main__":
    result = check()
    print(result)
