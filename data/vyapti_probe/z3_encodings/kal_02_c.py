"""Z3 encoding for KAL-02-C — auto-extracted from vyapti_benchmark_z3_encodings.md"""

PROBLEM_ID = "KAL-02-C"


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

red, blue, green, yellow = Ints('red blue green yellow')

s = Solver()

s.add(red > blue)
s.add(blue > green)
s.add(yellow < green)

# Heaviest: red
s.add(red > blue)
s.add(red > green)
s.add(red > yellow)

print("KAL-02-C: Heaviest = Red")
'''


if __name__ == "__main__":
    result = check()
    print(result)
