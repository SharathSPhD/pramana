"""Z3 encoding for KAL-05-C — auto-extracted from vyapti_benchmark_z3_encodings.md"""

PROBLEM_ID = "KAL-05-C"


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

build_year = Int('build_year')
ceiling_height = Int('ceiling_height')
compliant = Bool('compliant')

s = Solver()

s.add(build_year == 2018)
s.add(ceiling_height == 102)  # 8.5ft

# Pre-2020: 8ft = 96 inches sufficient
s.add(compliant == If(build_year > 2020, ceiling_height >= 108, ceiling_height >= 96))

s.check()
m = s.model()
print(f"KAL-05-C: Building W compliant? {m[compliant]}")  # True
'''


if __name__ == "__main__":
    result = check()
    print(result)
