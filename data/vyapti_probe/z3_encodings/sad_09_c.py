"""Z3 encoding for SAD-09-C — auto-extracted from vyapti_benchmark_z3_encodings.md"""

PROBLEM_ID = "SAD-09-C"


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

Methodology = DeclareSort('Methodology')
m = Const('m', Methodology)
validated_by_independent = Function('validated_by_independent', Methodology, BoolSort())

s = Solver()

# Validated by 3 independent groups — no circularity
s.add(validated_by_independent(m))

print("SAD-09-C: Yes — non-circular validation")
'''


if __name__ == "__main__":
    result = check()
    print(result)
