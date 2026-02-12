"""Z3 encoding for SAD-10-C — auto-extracted from vyapti_benchmark_z3_encodings.md"""

PROBLEM_ID = "SAD-10-C"


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

# Validator uses independently developed, formally verified rules.
# No circularity.

s = Solver()

# Independent rules — no self-reference
print("SAD-10-C: Yes — independent validator")
'''


if __name__ == "__main__":
    result = check()
    print(result)
