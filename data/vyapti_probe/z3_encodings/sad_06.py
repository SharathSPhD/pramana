"""Z3 encoding for SAD-06 — auto-extracted from vyapti_benchmark_z3_encodings.md"""

PROBLEM_ID = "SAD-06"


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

# Type-X if has Feature-P. Feature-P = feature of Type-X objects. Circular.
# Q: Is Q's classification valid? A: No (circular)

Entity = DeclareSort('Entity')
q = Const('q', Entity)
type_x = Function('type_x', Entity, BoolSort())
feature_p = Function('feature_p', Entity, BoolSort())

s = Solver()

# Circular: type_x(x) ↔ feature_p(x), feature_p defined as "feature of type_x"
s.add(ForAll([q], type_x(q) == feature_p(q)))
s.add(ForAll([q], feature_p(q) == type_x(q)))

# Can't determine classification
s.push()
s.add(type_x(q))
r1 = s.check()
s.pop()

s.push()
s.add(Not(type_x(q)))
r2 = s.check()
s.pop()

print(f"SAD-06: Classification circular: T {r1 == sat}, F {r2 == sat}")
print("SAD-06: No — circular definition")
'''


if __name__ == "__main__":
    result = check()
    print(result)
