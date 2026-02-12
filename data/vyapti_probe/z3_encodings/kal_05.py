"""Z3 encoding for KAL-05 — auto-extracted from vyapti_benchmark_z3_encodings.md"""

PROBLEM_ID = "KAL-05"


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

# Original 2015: 8ft ceiling. Amendment 2022: post-2020 buildings need 9ft.
# Building Z built 2023, 8.5ft. Q: Compliant? A: No

build_year = Int('build_year')
ceiling_height = Int('ceiling_height')
compliant = Bool('compliant')

s = Solver()

s.add(build_year == 2023)
s.add(ceiling_height == 85)  # 8.5ft in inches

# Post-2020: need 9ft = 108 inches
s.add(compliant == If(build_year > 2020, ceiling_height >= 108, ceiling_height >= 96))

s.check()
m = s.model()
print(f"KAL-05: Building Z compliant? {m[compliant]}")  # False
'''


if __name__ == "__main__":
    result = check()
    print(result)
