"""Vyapti benchmark evaluation runner.

Orchestrates the full 5-tier evaluation campaign across multiple models
on the 100-problem vyapti probe benchmark.
"""

import json
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Protocol


class ModelBackend(Protocol):
    """Protocol for model backends."""
    def generate(self, prompt: str, **kwargs: Any) -> str: ...


@dataclass
class TierResult:
    tier: int
    name: str
    passed: bool
    score: float
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProblemResult:
    problem_id: str
    category: str
    problem_type: str  # "probe" or "control"
    model_name: str
    raw_response: str
    response_length: int
    generation_time_ms: int
    tiers: list[TierResult] = field(default_factory=list)
    final_answer_correct: bool = False
    hetvabhasa_classification: str = ""
    hetvabhasa_used_fallback: bool = False


class VyaptiEvaluationRunner:
    """Run the full vyapti benchmark evaluation."""

    def __init__(self, config: dict, project_root: Path | None = None):
        self.config = config
        self.root = project_root or Path(__file__).resolve().parents[3]

        # Load benchmark data
        bench_path = self.root / config.get("benchmark_path", "data/vyapti_probe/problems.json")
        sol_path = self.root / config.get("solutions_path", "data/vyapti_probe/solutions.json")

        with open(bench_path) as f:
            self.problems = json.load(f)
        with open(sol_path) as f:
            solutions_list = json.load(f)
            self.solutions = {s["id"]: s for s in solutions_list}

        for problem in self.problems:
            pid = problem.get("id", "")
            if pid not in self.solutions:
                raise ValueError(f"Missing solution for problem {pid}")
            answer = self.solutions[pid].get("answer", "")
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError(f"Empty solution answer for problem {pid}")

        # Index problems
        self.problems_by_id = {p["id"]: p for p in self.problems}

        # Lazy-load scorers
        self._vyapti_scorer = None
        self._hetvabhasa_classifier = None

    @property
    def vyapti_scorer(self):
        if self._vyapti_scorer is None:
            from pramana.application.evaluation.vyapti_scorer import VyaptiScorer
            self._vyapti_scorer = VyaptiScorer()
        return self._vyapti_scorer

    @property
    def hetvabhasa_classifier(self):
        if self._hetvabhasa_classifier is None:
            from pramana.application.evaluation.hetvabhasa_classifier import HetvabhasaClassifier
            self._hetvabhasa_classifier = HetvabhasaClassifier()
        return self._hetvabhasa_classifier

    def build_prompt(self, problem: dict) -> str:
        """Build the prompt for a problem."""
        prompt_style = self.config.get("prompt_style", "direct")
        problem_text = problem["problem_text"]

        if prompt_style == "nyaya_template":
            return (
                f"### Problem:\n{problem_text}\n\n"
                "### Instructions:\n"
                "Analyze this problem using the 6-phase Nyaya methodology:\n"
                "1. Samshaya (Doubt Analysis)\n"
                "2. Pramana (Sources of Knowledge)\n"
                "3. Pancha Avayava (5-Member Syllogism)\n"
                "4. Tarka (Counterfactual Reasoning)\n"
                "5. Hetvabhasa (Fallacy Check)\n"
                "6. Nirnaya (Ascertainment)\n\n"
                "State your final answer clearly.\n"
            )
        elif prompt_style == "chain_of_thought":
            return (
                f"Problem: {problem_text}\n\n"
                "Think step by step. Check whether any universal rules hold without exception. "
                "Look for counterexamples before concluding. State your final answer.\n"
            )
        else:
            return f"Problem: {problem_text}\n\nWhat is the answer? Explain your reasoning.\n"

    def check_answer(self, response: str, problem: dict, solution: dict) -> bool:
        """Tier 1: Check if the final answer is correct."""
        correct = solution.get("answer", "").lower()
        resp_lower = response.lower()

        if not correct.strip():
            return False

        # Extract the answer orientation (yes/no/cannot determine)
        correct_is_no = correct.startswith("no")
        correct_is_yes = correct.startswith("yes")
        correct_is_undetermined = "cannot" in correct or "undetermined" in correct

        # Check response
        # Look for final answer patterns
        final_match = re.search(
            r"(?:final\s+answer|conclusion|therefore|thus|answer)\s*:?\s*(.+?)(?:\n|$)",
            resp_lower
        )
        answer_text = final_match.group(1) if final_match else resp_lower[:500]

        if correct_is_no:
            return bool(re.search(r"\bno\b", answer_text)) and not bool(re.search(r"\byes\b", answer_text[:80]))
        elif correct_is_yes:
            return bool(re.search(r"\byes\b", answer_text)) and not bool(re.search(r"\bno\b", answer_text[:80]))
        elif correct_is_undetermined:
            return any(p in answer_text for p in ["cannot", "undetermined", "insufficient", "impossible"])

        # Fallback: check for key phrases from correct answer
        key_phrases = re.findall(r'\b\w{4,}\b', correct)[:5]
        if not key_phrases:
            return False
        matches = sum(1 for p in key_phrases if p in resp_lower)
        return matches >= len(key_phrases) * 0.6

    def evaluate_problem(
        self, model_name: str, model_fn: Callable[[str], str], problem: dict
    ) -> ProblemResult:
        """Evaluate a single problem with all 5 tiers."""
        solution = self.solutions.get(problem["id"], {})
        prompt = self.build_prompt(problem)

        # Generate response
        start = time.time()
        try:
            response = model_fn(prompt)
        except Exception as e:
            response = f"[ERROR: {e}]"
        gen_time = int((time.time() - start) * 1000)

        result = ProblemResult(
            problem_id=problem["id"],
            category=problem["category"],
            problem_type=problem["type"],
            model_name=model_name,
            raw_response=response,
            response_length=len(response),
            generation_time_ms=gen_time,
        )

        # Tier 1: Answer correctness
        answer_correct = self.check_answer(response, problem, solution)
        result.final_answer_correct = answer_correct
        result.tiers.append(TierResult(
            tier=1, name="outcome", passed=answer_correct,
            score=1.0 if answer_correct else 0.0,
            details={"correct_answer": solution.get("answer", "")[:200]}
        ))

        # Tier 2: Structure (Pramana models only - check for 6-phase format)
        has_structure = any(
            phase in response for phase in
            ["Samshaya", "Pramana", "Pancha Avayava", "Tarka", "Hetvabhasa", "Nirnaya"]
        )
        result.tiers.append(TierResult(
            tier=2, name="structure", passed=has_structure,
            score=1.0 if has_structure else 0.0,
            details={"has_nyaya_structure": has_structure}
        ))

        # Tier 3: Vyapti explicitness
        vyapti_result = self.vyapti_scorer.score(response, problem, solution)
        result.tiers.append(TierResult(
            tier=3, name="vyapti_explicitness", passed=vyapti_result.correct,
            score=1.0 if vyapti_result.correct else 0.0,
            details={
                "stated": vyapti_result.stated,
                "correct": vyapti_result.correct,
                "negation_detected": vyapti_result.negation_detected,
                "vyapti_text": vyapti_result.vyapti_text[:200] if vyapti_result.vyapti_text else "",
            }
        ))

        # Tier 4: Z3 encoding execution (runs problem encoding, not model answer verification)
        z3_result = self._run_z3_check(problem["id"])
        result.tiers.append(TierResult(
            tier=4, name="z3_encoding_execution",
            passed=z3_result.get("success", False),
            score=1.0 if z3_result.get("success", False) else 0.0,
            details=z3_result,
        ))

        # Tier 5: Hetvabhasa classification (for incorrect answers)
        hclass = self.hetvabhasa_classifier.classify(
            problem, response, solution, answer_correct
        )
        result.hetvabhasa_classification = hclass.classified_type
        result.hetvabhasa_used_fallback = hclass.used_fallback
        result.tiers.append(TierResult(
            tier=5, name="hetvabhasa_classification",
            passed=hclass.matches_ground_truth,
            score=hclass.confidence,
            details={
                "classified_type": hclass.classified_type,
                "ground_truth_type": hclass.ground_truth_type,
                "confidence": hclass.confidence,
                "used_fallback": hclass.used_fallback,
                "evidence": hclass.evidence[:3],
            }
        ))

        return result

    def _run_z3_check(self, problem_id: str) -> dict:
        """Run the problem-specific Z3 encoding."""
        try:
            import sys
            root = str(self.root)
            if root not in sys.path:
                sys.path.insert(0, root)
            from data.vyapti_probe.z3_encodings import get_encoding
            mod = get_encoding(problem_id)
            if mod and hasattr(mod, "check"):
                result = mod.check()
                return {"success": True, "output": result.get("output", ""), "problem_id": problem_id}
            return {"success": False, "error": f"No Z3 encoding for {problem_id}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def evaluate_model(
        self, model_name: str, model_fn: Callable[[str], str],
        problem_ids: list[str] | None = None
    ) -> list[ProblemResult]:
        """Evaluate a model on all (or selected) problems."""
        problems = self.problems
        if problem_ids:
            problems = [p for p in self.problems if p["id"] in problem_ids]

        results = []
        for i, problem in enumerate(problems):
            print(f"  [{i+1}/{len(problems)}] {problem['id']}...", end=" ", flush=True)
            result = self.evaluate_problem(model_name, model_fn, problem)
            print(f"{'CORRECT' if result.final_answer_correct else 'WRONG'}")
            results.append(result)

        return results

    def run_all(self, models: dict[str, Callable[[str], str]]) -> dict[str, list[ProblemResult]]:
        """Run all models on all problems."""
        all_results = {}
        for model_name, model_fn in models.items():
            print(f"\n=== Evaluating: {model_name} ===")
            all_results[model_name] = self.evaluate_model(model_name, model_fn)
        return all_results

    def save_results(self, results: dict[str, list[ProblemResult]], output_dir: Path) -> None:
        """Save results to JSON files."""
        output_dir.mkdir(parents=True, exist_ok=True)

        for model_name, model_results in results.items():
            model_dir = output_dir / model_name.replace("/", "_").replace(" ", "_")
            model_dir.mkdir(parents=True, exist_ok=True)

            # Save per-problem results
            for r in model_results:
                with open(model_dir / f"{r.problem_id}.json", "w") as f:
                    json.dump(asdict(r), f, indent=2, default=str)

            # Save summary
            summary = self._compute_summary(model_name, model_results)
            with open(model_dir / "summary.json", "w") as f:
                json.dump(summary, f, indent=2)

        # Save cross-model summary
        cross_summary = {}
        for model_name, model_results in results.items():
            cross_summary[model_name] = self._compute_summary(model_name, model_results)
        with open(output_dir / "summary.json", "w") as f:
            json.dump(cross_summary, f, indent=2)

    def _compute_summary(self, model_name: str, results: list[ProblemResult]) -> dict:
        """Compute summary statistics for a model's results."""
        total = len(results)
        if total == 0:
            return {"model": model_name, "total": 0}

        correct = sum(1 for r in results if r.final_answer_correct)

        # By type
        probes = [r for r in results if r.problem_type == "probe"]
        controls = [r for r in results if r.problem_type == "control"]
        probe_correct = sum(1 for r in probes if r.final_answer_correct)
        control_correct = sum(1 for r in controls if r.final_answer_correct)

        # By category
        by_category = {}
        for cat in ["savyabhichara", "viruddha", "prakaranasama", "sadhyasama", "kalatita"]:
            cat_results = [r for r in results if r.category == cat]
            cat_probes = [r for r in cat_results if r.problem_type == "probe"]
            cat_controls = [r for r in cat_results if r.problem_type == "control"]
            by_category[cat] = {
                "total": len(cat_results),
                "correct": sum(1 for r in cat_results if r.final_answer_correct),
                "probe_correct": sum(1 for r in cat_probes if r.final_answer_correct),
                "probe_total": len(cat_probes),
                "control_correct": sum(1 for r in cat_controls if r.final_answer_correct),
                "control_total": len(cat_controls),
            }

        # Tier averages
        tier_scores = {f"tier_{i}": 0.0 for i in range(1, 6)}
        for r in results:
            for t in r.tiers:
                tier_scores[f"tier_{t.tier}"] += t.score
        tier_scores = {k: v / total for k, v in tier_scores.items()}

        # Hetvabhasa distribution
        hclass_dist = {}
        wrong = [r for r in results if not r.final_answer_correct]
        for r in wrong:
            htype = r.hetvabhasa_classification
            hclass_dist[htype] = hclass_dist.get(htype, 0) + 1

        return {
            "model": model_name,
            "total": total,
            "correct": correct,
            "accuracy": correct / total,
            "probe_accuracy": probe_correct / len(probes) if probes else 0,
            "control_accuracy": control_correct / len(controls) if controls else 0,
            "by_category": by_category,
            "tier_averages": tier_scores,
            "hetvabhasa_distribution": hclass_dist,
        }


def load_config(config_path: str | Path) -> dict:
    """Load evaluation config from YAML."""
    import yaml
    with open(config_path) as f:
        return yaml.safe_load(f)
