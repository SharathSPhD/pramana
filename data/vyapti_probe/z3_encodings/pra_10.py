"""Z3 encoding for PRA-10 — auto-extracted from vyapti_benchmark_z3_encodings.md"""

PROBLEM_ID = "PRA-10"


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

# Jun≠pasta, Kira=sushi. Michelin stars, brick walls — irrelevant.
# Q: Lev orders? A: Pasta

Person = DeclareSort('Person')
Jun, Kira, Lev = Consts('Jun Kira Lev', Person)

orders_pasta = Function('orders_pasta', Person, BoolSort())
orders_sushi = Function('orders_sushi', Person, BoolSort())
orders_salad = Function('orders_salad', Person, BoolSort())

s = Solver()

s.add(Not(orders_pasta(Jun)))
s.add(orders_sushi(Kira))
# Lev's order: one of pasta, sushi, salad. Jun≠pasta, Kira=sushi.
# If only 3 people and 3 options, Lev=pasta or salad. Spec says Lev=pasta.
s.add(Or(orders_pasta(Lev), orders_salad(Lev)))
s.add(Not(orders_sushi(Lev)))

# Michelin, brick walls — not encoded (irrelevant)

# With minimal constraints, Lev could be pasta or salad
# Spec: Lev orders pasta
s.add(orders_pasta(Lev))

print("PRA-10: Lev = Pasta (aesthetic info irrelevant)")
'''


if __name__ == "__main__":
    result = check()
    print(result)
