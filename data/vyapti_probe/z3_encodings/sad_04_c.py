"""Z3 encoding for SAD-04-C — auto-extracted from vyapti_benchmark_z3_encodings.md"""

PROBLEM_ID = "SAD-04-C"


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

User = DeclareSort('User')
alice, bob, charlie, dana = Consts('alice bob charlie dana', User)
on_list = Function('on_list', User, BoolSort())

s = Solver()

# List = {Alice, Bob, Charlie} (maintained by security team)
s.add(on_list(alice))
s.add(on_list(bob))
s.add(on_list(charlie))
s.add(Not(on_list(dana)))

# Authorized = on list
s.push()
s.add(on_list(dana))
print(f"SAD-04-C: Is Dana on list? {s.check() == sat}")  # UNSAT
s.pop()

print("SAD-04-C: No — Dana not authorized")
'''


if __name__ == "__main__":
    result = check()
    print(result)
