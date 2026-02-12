#!/usr/bin/env python3
"""Rescore existing Vyapti result artifacts with updated logic."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pramana.benchmarks.vyapti_runner import ProblemResult, TierResult, VyaptiEvaluationRunner


def _to_problem_result(data: dict) -> ProblemResult:
    tiers = [
        TierResult(
            tier=t.get("tier", 0),
            name=t.get("name", ""),
            passed=t.get("passed", False),
            score=t.get("score", 0.0),
            details=t.get("details", {}),
        )
        for t in data.get("tiers", [])
    ]
    return ProblemResult(
        problem_id=data.get("problem_id", ""),
        category=data.get("category", ""),
        problem_type=data.get("problem_type", ""),
        model_name=data.get("model_name", ""),
        raw_response=data.get("raw_response", ""),
        response_length=data.get("response_length", 0),
        generation_time_ms=data.get("generation_time_ms", 0),
        tiers=tiers,
        final_answer_correct=data.get("final_answer_correct", False),
        hetvabhasa_classification=data.get("hetvabhasa_classification", ""),
        hetvabhasa_used_fallback=data.get("hetvabhasa_used_fallback", False),
    )


def _update_or_append_tier(result_data: dict, tier_obj: TierResult) -> None:
    tiers = result_data.get("tiers", [])
    for idx, tier in enumerate(tiers):
        if tier.get("tier") == tier_obj.tier:
            tiers[idx] = asdict(tier_obj)
            result_data["tiers"] = tiers
            return
    tiers.append(asdict(tier_obj))
    result_data["tiers"] = tiers


def main() -> int:
    config_path = PROJECT_ROOT / "configs" / "vyapti_eval.yaml"
    config = yaml.safe_load(config_path.read_text())
    results_dir = PROJECT_ROOT / config.get("results_path", "data/vyapti_probe/results_real")
    models_config = config.get("models", {})

    runner = VyaptiEvaluationRunner(config, project_root=PROJECT_ROOT)
    problems_by_id = runner.problems_by_id

    total_files = 0
    total_changed = 0
    cross_summary: dict[str, dict] = {}

    for model_name in models_config:
        model_dir = results_dir / model_name
        if not model_dir.exists():
            print(f"[WARN] missing model directory: {model_dir}")
            continue

        model_results: list[ProblemResult] = []
        changed_for_model = 0
        problem_files = sorted(
            path for path in model_dir.glob("*.json")
            if path.name not in {"summary.json"}
        )

        for result_file in problem_files:
            total_files += 1
            data = json.loads(result_file.read_text())
            problem_id = data.get("problem_id") or result_file.stem
            problem = problems_by_id.get(problem_id)
            solution = runner.solutions.get(problem_id, {})
            if not problem:
                print(f"[WARN] skipping unknown problem id {problem_id} in {result_file.name}")
                continue

            old_correct = data.get("final_answer_correct", False)
            old_class = data.get("hetvabhasa_classification", "")
            old_fallback = data.get("hetvabhasa_used_fallback", False)

            response = data.get("raw_response", "")
            answer_correct = runner.check_answer(response, problem, solution)
            hclass = runner.hetvabhasa_classifier.classify(problem, response, solution, answer_correct)
            z3_result = runner._run_z3_check(problem_id)

            data["model_name"] = model_name
            data["final_answer_correct"] = answer_correct
            data["hetvabhasa_classification"] = hclass.classified_type
            data["hetvabhasa_used_fallback"] = hclass.used_fallback

            tier1 = TierResult(
                tier=1,
                name="outcome",
                passed=answer_correct,
                score=1.0 if answer_correct else 0.0,
                details={"correct_answer": solution.get("answer", "")[:200]},
            )
            tier4 = TierResult(
                tier=4,
                name="z3_encoding_execution",
                passed=z3_result.get("success", False),
                score=1.0 if z3_result.get("success", False) else 0.0,
                details=z3_result,
            )
            tier5 = TierResult(
                tier=5,
                name="hetvabhasa_classification",
                passed=hclass.matches_ground_truth,
                score=hclass.confidence,
                details={
                    "classified_type": hclass.classified_type,
                    "ground_truth_type": hclass.ground_truth_type,
                    "confidence": hclass.confidence,
                    "used_fallback": hclass.used_fallback,
                    "evidence": hclass.evidence[:3],
                },
            )
            _update_or_append_tier(data, tier1)
            _update_or_append_tier(data, tier4)
            _update_or_append_tier(data, tier5)

            if (
                old_correct != data["final_answer_correct"]
                or old_class != data["hetvabhasa_classification"]
                or old_fallback != data["hetvabhasa_used_fallback"]
            ):
                changed_for_model += 1

            result_file.write_text(json.dumps(data, indent=2, default=str))
            model_results.append(_to_problem_result(data))

        if model_results:
            summary = runner._compute_summary(model_name, model_results)
            (model_dir / "summary.json").write_text(json.dumps(summary, indent=2))
            cross_summary[model_name] = summary

        total_changed += changed_for_model
        print(f"{model_name}: rescored {len(model_results)} files, changed {changed_for_model}")

    (results_dir / "summary.json").write_text(json.dumps(cross_summary, indent=2))
    print(f"\nRescoring complete: {total_files} files processed, {total_changed} files changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
