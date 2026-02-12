#!/usr/bin/env python3
"""Validate consistency of Vyapti benchmark artifacts."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

VALID_CATEGORIES = {
    "savyabhichara",
    "viruddha",
    "prakaranasama",
    "sadhyasama",
    "kalatita",
}


def normalize_answer(answer: str) -> str:
    """Normalize a free-form answer to yes/no/undetermined/other."""
    lowered = answer.strip().lower()
    if lowered.startswith("yes"):
        return "yes"
    if lowered.startswith("no"):
        return "no"
    if any(token in lowered for token in ("cannot", "undetermined", "insufficient", "impossible")):
        return "undetermined"
    return "other"


def extract_z3_expected_answer(module) -> str | None:
    """Extract expected answer from encoding comments, if present."""
    code = getattr(module, "_CODE", "")
    match = re.search(r"Correct answer:\s*([^\n]+)", code)
    if not match:
        return None
    return normalize_answer(match.group(1))


def main() -> int:
    problems_path = PROJECT_ROOT / "data" / "vyapti_probe" / "problems.json"
    solutions_path = PROJECT_ROOT / "data" / "vyapti_probe" / "solutions.json"

    problems = json.loads(problems_path.read_text())
    solutions = json.loads(solutions_path.read_text())
    solutions_by_id = {entry.get("id", ""): entry for entry in solutions}

    errors: list[str] = []
    warnings: list[str] = []

    problem_ids = [problem.get("id", "") for problem in problems]
    duplicate_problem_ids = {pid for pid in problem_ids if problem_ids.count(pid) > 1}
    if duplicate_problem_ids:
        errors.append(f"Duplicate problem IDs: {sorted(duplicate_problem_ids)}")

    try:
        from data.vyapti_probe.z3_encodings import get_encoding
    except Exception as exc:  # pragma: no cover - runtime environment dependent
        errors.append(f"Unable to import z3 encodings registry: {exc}")
        get_encoding = None

    for problem in problems:
        pid = problem.get("id", "")
        category = problem.get("category", "")

        if category not in VALID_CATEGORIES:
            errors.append(f"{pid}: invalid category '{category}'")

        solution = solutions_by_id.get(pid)
        if solution is None:
            errors.append(f"{pid}: missing solution entry")
            continue

        answer = solution.get("answer", "")
        if not isinstance(answer, str) or not answer.strip():
            errors.append(f"{pid}: empty solution.answer")
            continue

        if category == "savyabhichara":
            if get_encoding is None:
                continue
            try:
                encoding_module = get_encoding(pid)
                check_result = encoding_module.check()
                output = check_result.get("output", "")
                if not output.strip():
                    warnings.append(f"{pid}: encoding executed but produced empty output")
            except Exception as exc:
                errors.append(f"{pid}: z3 encoding execution failed: {exc}")
                continue

            expected = extract_z3_expected_answer(encoding_module)
            if expected is not None:
                observed = normalize_answer(answer)
                if expected == "other":
                    warnings.append(
                        f"{pid}: unable to normalize Z3 annotation to yes/no/undetermined"
                    )
                elif observed == "other":
                    warnings.append(
                        f"{pid}: unable to normalize solution answer to yes/no/undetermined"
                    )
                elif expected != observed:
                    errors.append(
                        f"{pid}: solution answer '{observed}' does not match Z3 annotation '{expected}'"
                    )
            else:
                warnings.append(f"{pid}: no 'Correct answer' annotation found in encoding")

    extra_solution_ids = set(solutions_by_id) - set(problem_ids)
    if extra_solution_ids:
        warnings.append(f"Solutions without matching problems: {sorted(extra_solution_ids)}")

    print("Vyapti benchmark validation report")
    print(f"- Problems: {len(problems)}")
    print(f"- Solutions: {len(solutions)}")
    print(f"- Errors: {len(errors)}")
    print(f"- Warnings: {len(warnings)}")

    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"  - {warning}")

    if errors:
        print("\nErrors:")
        for error in errors:
            print(f"  - {error}")
        return 1

    print("\nValidation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
