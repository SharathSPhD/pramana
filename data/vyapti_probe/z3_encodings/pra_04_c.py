"""Z3 encoding for PRA-04-C — auto-extracted from vyapti_benchmark_z3_encodings.md"""

PROBLEM_ID = "PRA-04-C"


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

s = Solver()

time_route = 2 + 3 + 1
print(f"PRA-04-C: A to D time = {time_route}hrs")
'''


if __name__ == "__main__":
    result = check()
    print(result)
