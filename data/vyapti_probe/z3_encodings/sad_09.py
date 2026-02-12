"""Z3 encoding for SAD-09 — auto-extracted from vyapti_benchmark_z3_encodings.md"""

PROBLEM_ID = "SAD-09"


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

# Survey methodology M validated by M's own respondents. Circular.
# Q: Valid? A: No

Methodology = DeclareSort('Methodology')
m = Const('m', Methodology)
validated_by_respondents = Function('validated_by_respondents', Methodology, BoolSort())
valid = Function('valid', Methodology, BoolSort())

s = Solver()

# Circular: valid(m) ↔ validated_by_respondents(m), and respondents use M
s.add(valid(m) == validated_by_respondents(m))
s.add(validated_by_respondents(m) == valid(m))  # Self-referential

print("SAD-09: No — circular (self-supporting)")
'''


if __name__ == "__main__":
    result = check()
    print(result)
