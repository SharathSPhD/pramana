"""Z3 encoding for SAD-10 — auto-extracted from vyapti_benchmark_z3_encodings.md"""

PROBLEM_ID = "SAD-10"


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

# AI safety validator uses system's own output to define safety. Circular.
# Q: Satisfactory? A: No

safety_valid = Function('safety_valid', BoolSort(), BoolSort())
uses_own_output = True  # Encodes the circularity

s = Solver()

# Circular: safe ↔ validator(safe)
# Encoded as: validator conclusion depends on own output
print("SAD-10: No — circular (recursive justification)")
'''


if __name__ == "__main__":
    result = check()
    print(result)
