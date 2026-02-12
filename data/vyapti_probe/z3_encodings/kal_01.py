"""Z3 encoding for KAL-01 — auto-extracted from vyapti_benchmark_z3_encodings.md"""

PROBLEM_ID = "KAL-01"


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

# Priority levels: higher number = higher priority
general_days = 15
dept_days = 20
individual_days = 10

employee_id = Int('employee_id')
is_engineering = Bool('is_engineering')
is_4472 = Bool('is_4472')
vacation_days = Int('vacation_days')

s = Solver()

s.add(employee_id == 4472)
s.add(is_engineering == True)
s.add(is_4472 == True)

# Rules in priority order (later overrides earlier)
# Default: 15
# Engineering override: 20
# Individual exception for 4472: 10
s.add(If(is_4472, vacation_days == individual_days,
         If(is_engineering, vacation_days == dept_days,
            vacation_days == general_days)))

s.check()
m = s.model()
print(f"KAL-01: Vacation days for #4472: {m[vacation_days]}")  # 10
'''


if __name__ == "__main__":
    result = check()
    print(result)
