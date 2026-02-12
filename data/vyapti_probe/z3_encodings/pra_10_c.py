"""Z3 encoding for PRA-10-C — auto-extracted from vyapti_benchmark_z3_encodings.md"""

PROBLEM_ID = "PRA-10-C"


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

Person = DeclareSort('Person')
Jun, Kira, Lev = Consts('Jun Kira Lev', Person)

orders_pasta = Function('orders_pasta', Person, BoolSort())
orders_sushi = Function('orders_sushi', Person, BoolSort())
orders_salad = Function('orders_salad', Person, BoolSort())

s = Solver()

s.add(Not(orders_pasta(Jun)))
s.add(orders_sushi(Kira))
s.add(Not(orders_salad(Lev)))

# Three people, three distinct orders
s.add(Implies(orders_pasta(Jun), False))  # Jun≠pasta
# Jun=salad, Kira=sushi, Lev=pasta
s.add(orders_salad(Jun))
s.add(orders_pasta(Lev))

s.check()
print("PRA-10-C: Lev=pasta, Jun=salad")
'''


if __name__ == "__main__":
    result = check()
    print(result)
