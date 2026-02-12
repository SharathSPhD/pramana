"""Z3 encoding for SAD-04 — auto-extracted from vyapti_benchmark_z3_encodings.md"""

PROBLEM_ID = "SAD-04"


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

# Authorized ↔ on list. List = authorized users. Circular.
# Q: Is User Y authorized? A: Cannot determine

User = DeclareSort('User')
y = Const('y', User)
on_list = Function('on_list', User, BoolSort())
authorized = Function('authorized', User, BoolSort())

s = Solver()

# Circular: authorized ↔ on_list, list defined by authorized
s.add(ForAll([y], authorized(y) == on_list(y)))
# List = authorized users — so on_list(y) ↔ authorized(y). Circular.
s.add(ForAll([y], on_list(y) == authorized(y)))

# Underdetermined
s.push()
s.add(authorized(y))
r1 = s.check()
s.pop()

s.push()
s.add(Not(authorized(y)))
r2 = s.check()
s.pop()

print(f"SAD-04: Y authorized can be T: {r1 == sat}, F: {r2 == sat}")
print("SAD-04: Cannot determine (circular)")
'''


if __name__ == "__main__":
    result = check()
    print(result)
