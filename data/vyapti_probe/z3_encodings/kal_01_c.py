"""Z3 encoding for KAL-01-C — auto-extracted from vyapti_benchmark_z3_encodings.md"""

PROBLEM_ID = "KAL-01-C"


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

employee_id = Int('employee_id')
is_engineering = Bool('is_engineering')
is_4472 = Bool('is_4472')
vacation_days = Int('vacation_days')

s = Solver()

s.add(employee_id == 5501)
s.add(is_engineering == True)
s.add(is_4472 == False)

s.add(If(is_4472, vacation_days == 10,
         If(is_engineering, vacation_days == 20,
            vacation_days == 15)))

s.check()
m = s.model()
print(f"KAL-01-C: Vacation days for #5501: {m[vacation_days]}")  # 20
'''


if __name__ == "__main__":
    result = check()
    print(result)
