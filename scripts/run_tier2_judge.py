"""Run Tier 2 LLM judge on Stage 0 outputs."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from pramana.application.evaluation.llm_judge import Tier2LLMJudgeHandler
from pramana.config.settings import PramanaSettings
from pramana.infrastructure.llm import LLMClientError, create_llm_client


@dataclass
class JudgeExample:
    """Minimal example wrapper for Tier 2 evaluation."""

    problem: str


def _load_results(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _write_results(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main() -> None:
    results_path = os.getenv("RESULTS_FILE", "results/stage_0_final_validation.json")
    output_path = os.getenv("OUTPUT_FILE", "results/stage_0_tier2_judge.json")

    settings = PramanaSettings()

    try:
        llm_client = create_llm_client(settings)
    except LLMClientError as exc:
        raise SystemExit(f"Tier 2 judge configuration error: {exc}") from exc

    handler = Tier2LLMJudgeHandler(llm_client=llm_client)
    data = _load_results(results_path)

    results: list[dict[str, Any]] = []
    passed = 0
    total = 0
    score_sum = 0.0

    for item in data.get("results", []):
        total += 1
        example = JudgeExample(problem=item.get("problem", ""))
        output = item.get("generated_output", "")
        tier_result = handler.evaluate(example, output)

        if tier_result.passed:
            passed += 1
        score_sum += tier_result.score

        results.append(
            {
                "example_id": item.get("example_id"),
                "tier2": {
                    "passed": tier_result.passed,
                    "score": tier_result.score,
                    "details": tier_result.details,
                    "errors": tier_result.errors,
                },
            }
        )

    summary = {
        "total_examples": total,
        "passed": passed,
        "pass_rate": passed / total if total else 0.0,
        "mean_score": score_sum / total if total else 0.0,
    }

    payload = {
        "evaluation_timestamp": datetime.utcnow().isoformat(),
        "provider": settings.llm_provider,
        "model": settings.openai_judge_model
        if settings.llm_provider.lower() == "openai"
        else settings.anthropic_judge_model,
        "input_results_file": results_path,
        "results": results,
        "summary": summary,
    }

    _write_results(output_path, payload)
    print(f"Tier 2 judge results saved to {output_path}")


if __name__ == "__main__":
    main()
