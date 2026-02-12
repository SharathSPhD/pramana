"""Z3 encoding for UTILITY — auto-extracted from vyapti_benchmark_z3_encodings.md"""

PROBLEM_ID = "UTILITY"


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

def check_vyapti(premises, hetu_predicate, sadhya_predicate, domain):
    """
    Generic vyāpti checker.
    
    Args:
        premises: list of Z3 constraints (the problem's stated facts)
        hetu_predicate: function(entity) -> Z3 Bool (the proposed reason)
        sadhya_predicate: function(entity) -> Z3 Bool (the proposed conclusion)
        domain: list of Z3 constants (entities to check)
    
    Returns:
        dict with:
            - holds: bool (does vyāpti hold universally?)
            - counterexamples: list (entities where hetu but not sadhya)
            - hetvabhasa_type: str (if fails, which fallacy)
    """
    s = Solver()
    s.add(*premises)
    
    counterexamples = []
    
    for entity in domain:
        s.push()
        s.add(hetu_predicate(entity))
        s.add(Not(sadhya_predicate(entity)))
        
        if s.check() == sat:
            counterexamples.append(entity)
        s.pop()
    
    if not counterexamples:
        return {"holds": True, "counterexamples": [], "hetvabhasa_type": None}
    
    # Classify hetvabhasa type
    # Check: does hetu appear in vipaksha (savyabhichara)?
    hetu_in_vipaksha = len(counterexamples) > 0
    hetu_in_paksha = False
    for entity in domain:
        s.push()
        s.add(hetu_predicate(entity))
        s.add(sadhya_predicate(entity))
        if s.check() == sat:
            hetu_in_paksha = True
        s.pop()
    
    if hetu_in_paksha and hetu_in_vipaksha:
        hetvabhasa = "savyabhichara"  # Hetu in both paksha and vipaksha
    elif not hetu_in_paksha and hetu_in_vipaksha:
        hetvabhasa = "viruddha"  # Hetu only in vipaksha
    else:
        hetvabhasa = "unclassified"
    
    return {
        "holds": False,
        "counterexamples": [str(e) for e in counterexamples],
        "hetvabhasa_type": hetvabhasa
    }
'''


if __name__ == "__main__":
    result = check()
    print(result)
