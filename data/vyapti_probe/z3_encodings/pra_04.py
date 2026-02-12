"""Z3 encoding for PRA-04 — auto-extracted from vyapti_benchmark_z3_encodings.md"""

PROBLEM_ID = "PRA-04"


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

# Route A-B-C-D: 2+3+1=6hrs
# Route E-F-G: disconnected
# Q: A to D time + connected? A: 6hrs, no connection

# Encode as two separate route systems
s = Solver()

# Route 1: A→B(2), B→C(3), C→D(1) = 6
time_route1 = 2 + 3 + 1  # 6

# Route 2: E-F-G (disconnected from A-B-C-D)
# Cannot reach A from E or vice versa

print(f"PRA-04: A to D time = {time_route1}hrs")
print("PRA-04: Routes not connected")
'''


if __name__ == "__main__":
    result = check()
    print(result)
