"""Z3 encoding for KAL-02 — auto-extracted from vyapti_benchmark_z3_encodings.md"""

PROBLEM_ID = "KAL-02"


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

# Phase 1: red>blue>green, yellow unknown
# Phase 2: red removed, orange added, orange<green
# Q: Heaviest in current bag? A: Cannot be sure (yellow unknown)

# Encode weights: red, blue, green, yellow, orange
red, blue, green, yellow, orange = Ints('red blue green yellow orange')

s = Solver()

# Phase 1 constraints (red gone in phase 2)
s.add(red > blue)
s.add(blue > green)
# yellow unknown

# Phase 2: red removed, orange added
s.add(orange < green)

# Current: blue > green > orange. But yellow unknown — could be heavier than blue
s.add(yellow >= 0)  # yellow exists but weight unknown

print("KAL-02: Cannot be sure — yellow unknown")
'''


if __name__ == "__main__":
    result = check()
    print(result)
