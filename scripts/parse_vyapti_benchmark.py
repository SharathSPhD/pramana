#!/usr/bin/env python3
"""
Phase 1a+1c: Parse vyapti benchmark markdown files into structured JSON.

Reads:
  - docs/vyapti/vyapti_probe_benchmark.md (50 probes)
  - docs/vyapti/vyapti_benchmark_controls.md (50 controls)
  - docs/vyapti/vyapti_benchmark_solutions.md (100 solutions)

Writes:
  - data/vyapti_probe/problems.json (100 problem objects)
  - data/vyapti_probe/solutions.json (100 solution objects)
"""

import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS = PROJECT_ROOT / "docs" / "vyapti"
OUT = PROJECT_ROOT / "data" / "vyapti_probe"


# ─── Category mapping ───

CATEGORY_MAP = {
    "SAV": "savyabhichara",
    "VIR": "viruddha",
    "PRA": "prakaranasama",
    "SAD": "sadhyasama",
    "KAL": "kalatita",
}

CATEGORY_FULL = {
    "savyabhichara": "Savyabhichāra (Erratic Middle Term)",
    "viruddha": "Viruddha (Contradictory Middle Term)",
    "prakaranasama": "Prakaraṇasama (Irrelevant Middle Term)",
    "sadhyasama": "Sādhyasama (Circular/Question-Begging)",
    "kalatita": "Kālātīta (Temporally Invalid)",
}


def clean(text: str) -> str:
    """Strip and clean whitespace."""
    return re.sub(r"\n{3,}", "\n\n", text.strip())


# ═══════════════════════════════════════════════════════════════
# 1. PARSE PROBES
# ═══════════════════════════════════════════════════════════════

def parse_probes(path: Path) -> list[dict]:
    """Parse vyapti_probe_benchmark.md into structured problem objects."""
    text = path.read_text(encoding="utf-8")

    # Split on problem headers: ### SAV-01 | Constraint Satisfaction | Difficulty: 2
    pattern = r"###\s+((?:SAV|VIR|PRA|SAD|KAL)-\d+)\s*\|\s*(.+?)\s*\|\s*Difficulty:\s*(\d+)"
    splits = re.split(f"({pattern})", text)

    problems = []
    i = 1  # skip preamble
    while i < len(splits):
        # splits[i] is the full match, [i+1]=id, [i+2]=logic_type, [i+3]=difficulty
        full_header = splits[i]
        prob_id = splits[i + 1]
        logic_type = splits[i + 2].strip()
        difficulty = int(splits[i + 3])

        # Body is the next non-header chunk
        body = splits[i + 4] if (i + 4) < len(splits) else ""

        cat_code = prob_id.split("-")[0]
        category = CATEGORY_MAP.get(cat_code, "unknown")

        # Extract fields from body
        problem_text = extract_field(body, "Problem")
        correct_answer = extract_field(body, "Correct Answer")
        trap_answer = extract_field(body, "Trap Answer")
        vyapti_under_test = extract_field(body, "Vyāpti Under Test")
        why_it_fails = extract_field(body, "Why It Fails")
        matched_control = extract_field(body, "Matched Control")

        # Handle revised problems
        revised = extract_field(body, "Correct Answer.*?\\(revised\\)")
        if revised:
            correct_answer = revised

        problems.append({
            "id": prob_id,
            "category": category,
            "type": "probe",
            "difficulty": difficulty,
            "logic_type": logic_type.lower().replace(" ", "_"),
            "problem_text": clean(problem_text) if problem_text else "",
            "correct_answer": clean(correct_answer) if correct_answer else "",
            "trap_answer": clean(trap_answer) if trap_answer else "",
            "vyapti_under_test": clean(vyapti_under_test) if vyapti_under_test else "",
            "why_it_fails": clean(why_it_fails) if why_it_fails else "",
            "matched_pair": f"{prob_id}-C",
            "hetvabhasa_type": category,
        })

        i += 5

    return problems


def extract_field(body: str, field_name: str) -> str | None:
    """Extract a **Field**: value block from markdown body."""
    # Match **Field**: or **Field** (revised): etc.
    pattern = rf"\*\*{field_name}\*\*(?:\s*\(revised\))?\s*:\s*(.*?)(?=\n\*\*[A-Z]|\n###|\n---|\n#\s|\n```|\Z)"
    match = re.search(pattern, body, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


# ═══════════════════════════════════════════════════════════════
# 2. PARSE CONTROLS
# ═══════════════════════════════════════════════════════════════

def parse_controls(path: Path) -> list[dict]:
    """Parse vyapti_benchmark_controls.md into structured problem objects."""
    text = path.read_text(encoding="utf-8")

    pattern = r"###\s+((?:SAV|VIR|PRA|SAD|KAL)-\d+-C)\s*\|\s*(.+?)\s*\|\s*Difficulty:\s*(\d+)"
    splits = re.split(f"({pattern})", text)

    problems = []
    i = 1
    while i < len(splits):
        full_header = splits[i]
        prob_id = splits[i + 1]
        logic_type = splits[i + 2].strip()
        difficulty = int(splits[i + 3])
        body = splits[i + 4] if (i + 4) < len(splits) else ""

        cat_code = prob_id.split("-")[0]
        category = CATEGORY_MAP.get(cat_code, "unknown")

        # Controls may have revised problems
        problem_text = extract_field(body, "Problem")
        # Check for revised version
        revised_match = re.search(
            r"\*\*Problem\*\*\s*\(revised\)\s*:\s*(.*?)(?=\n\*\*[A-Z]|\n###|\n---|\Z)",
            body, re.DOTALL
        )
        if revised_match:
            problem_text = revised_match.group(1).strip()

        correct_answer = extract_field(body, "Correct Answer")
        key_diff = extract_field(body, "Key Difference from Probe")
        note = extract_field(body, "Note")

        # Some controls have the correct answer inline in a different place
        # Also check for "Correct Answer" in revised block
        revised_answer = None
        rev_a = re.search(
            r"\*\*Correct Answer\*\*\s*\(revised\)\s*:\s*(.*?)(?=\n\*\*|\n###|\n---|\Z)",
            body, re.DOTALL
        )
        if rev_a:
            revised_answer = rev_a.group(1).strip()

        probe_id = prob_id.replace("-C", "")

        problems.append({
            "id": prob_id,
            "category": category,
            "type": "control",
            "difficulty": difficulty,
            "logic_type": logic_type.lower().replace(" ", "_"),
            "problem_text": clean(problem_text) if problem_text else "",
            "correct_answer": clean(revised_answer or correct_answer or ""),
            "trap_answer": "",  # Controls don't have trap answers
            "vyapti_under_test": "",
            "why_it_fails": "",
            "matched_pair": probe_id,
            "hetvabhasa_type": category,
            "key_difference": clean(key_diff) if key_diff else "",
        })

        i += 5

    return problems


# ═══════════════════════════════════════════════════════════════
# 3. PARSE SOLUTIONS
# ═══════════════════════════════════════════════════════════════

def parse_solutions(path: Path) -> list[dict]:
    """Parse vyapti_benchmark_solutions.md into structured solution objects."""
    text = path.read_text(encoding="utf-8")

    solutions = []

    # Parse the quick reference table first for IDs
    # Then parse detailed solutions

    # Split on ### headers like "### SAV-01 — Engineering Remote Work"
    # or "### SAV-01-C — Control: Engineering Remote (Rule-Based)"
    detail_pattern = r"###\s+((?:SAV|VIR|PRA|SAD|KAL)-\d+(?:-C)?)\s*[—–-]\s*(.+?)(?:\n)"
    parts = re.split(f"({detail_pattern})", text)

    i = 1
    while i < len(parts):
        full_match = parts[i]
        sol_id = parts[i + 1]
        title = parts[i + 2].strip()
        body = parts[i + 3] if (i + 3) < len(parts) else ""

        answer = extract_solution_field(body, "Answer")
        justification = extract_solution_field(body, "Logical Justification")
        vyapti_status = extract_solution_field(body, "Vyāpti Status")
        counterexample = extract_solution_field(body, "Counterexample")
        hetvabhasa = extract_solution_field(body, "Hetvābhāsa Classification")
        z3_note = extract_solution_field(body, "Z3 Verification")

        # Determine if vyapti holds
        vyapti_holds = None
        if vyapti_status:
            vs_lower = vyapti_status.lower()
            if "holds" in vs_lower and "fail" not in vs_lower:
                vyapti_holds = True
            elif "fail" in vs_lower:
                vyapti_holds = False
            elif "n/a" in vs_lower or "not applicable" in vs_lower:
                vyapti_holds = None

        solutions.append({
            "id": sol_id,
            "title": title,
            "answer": clean(answer) if answer else "",
            "justification": clean(justification) if justification else "",
            "vyapti_holds": vyapti_holds,
            "vyapti_status": clean(vyapti_status) if vyapti_status else "",
            "counterexample": clean(counterexample) if counterexample else "",
            "hetvabhasa_type": clean(hetvabhasa) if hetvabhasa else "",
            "z3_verification": clean(z3_note) if z3_note else "",
        })

        i += 4

    return solutions


def extract_solution_field(body: str, field_name: str) -> str | None:
    """Extract - **Field**: value from a solution block."""
    pattern = rf"-\s*\*\*{field_name}\*\*\s*:\s*(.*?)(?=\n-\s*\*\*|\n###|\n---|\Z)"
    match = re.search(pattern, body, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    OUT.mkdir(parents=True, exist_ok=True)

    print("Parsing probes...")
    probes = parse_probes(DOCS / "vyapti_probe_benchmark.md")
    print(f"  Found {len(probes)} probes")
    for cat in CATEGORY_MAP.values():
        count = sum(1 for p in probes if p["category"] == cat)
        print(f"    {cat}: {count}")

    print("Parsing controls...")
    controls = parse_controls(DOCS / "vyapti_benchmark_controls.md")
    print(f"  Found {len(controls)} controls")

    print("Parsing solutions...")
    solutions = parse_solutions(DOCS / "vyapti_benchmark_solutions.md")
    print(f"  Found {len(solutions)} solutions")
    probe_sols = sum(1 for s in solutions if "-C" not in s["id"])
    control_sols = sum(1 for s in solutions if "-C" in s["id"])
    print(f"    Probe solutions: {probe_sols}")
    print(f"    Control solutions: {control_sols}")

    # Combine problems
    all_problems = probes + controls
    print(f"\nTotal problems: {len(all_problems)}")

    # Write problems.json
    problems_path = OUT / "problems.json"
    with open(problems_path, "w", encoding="utf-8") as f:
        json.dump(all_problems, f, indent=2, ensure_ascii=False)
    print(f"Wrote {problems_path}")

    # Write solutions.json
    solutions_path = OUT / "solutions.json"
    with open(solutions_path, "w", encoding="utf-8") as f:
        json.dump(solutions, f, indent=2, ensure_ascii=False)
    print(f"Wrote {solutions_path}")

    # Validation: check all probes have solutions
    prob_ids = {p["id"] for p in all_problems}
    sol_ids = {s["id"] for s in solutions}
    missing_solutions = prob_ids - sol_ids
    extra_solutions = sol_ids - prob_ids
    if missing_solutions:
        print(f"\nWARNING: Problems without solutions: {sorted(missing_solutions)}")
    if extra_solutions:
        print(f"\nWARNING: Solutions without problems: {sorted(extra_solutions)}")

    # Stats
    print("\n=== Summary ===")
    print(f"Problems: {len(all_problems)} ({len(probes)} probes + {len(controls)} controls)")
    print(f"Solutions: {len(solutions)}")
    print(f"Categories: {', '.join(CATEGORY_MAP.values())}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
