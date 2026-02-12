#!/usr/bin/env python3
"""Phase 3: Run the vyapti benchmark evaluation campaign.

Usage:
    # Full evaluation (requires GPU + models):
    python scripts/run_vyapti_evaluation.py --config configs/vyapti_eval.yaml

    # Simulation mode (generates synthetic results for pipeline testing):
    python scripts/run_vyapti_evaluation.py --simulate

    # Evaluate a single model:
    python scripts/run_vyapti_evaluation.py --config configs/vyapti_eval.yaml --model stage1_pramana
"""

import argparse
import json
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pramana.benchmarks.vyapti_runner import (
    VyaptiEvaluationRunner,
    ProblemResult,
    TierResult,
)


# ═══════════════════════════════════════════════════════════════
# Simulation mode: Generate realistic synthetic results
# ═══════════════════════════════════════════════════════════════

# Expected accuracy profiles based on the vyapti hypothesis
# Format: {model: {probe_acc: float, control_acc: float, variation_by_category: dict}}
EXPECTED_PROFILES = {
    "llama_3b_base": {
        "probe_acc": 0.30,
        "control_acc": 0.72,
        "category_multiplier": {
            "savyabhichara": 0.9,  # Most susceptible to pattern following
            "viruddha": 1.1,
            "prakaranasama": 0.8,
            "sadhyasama": 1.0,
            "kalatita": 1.2,
        },
    },
    "deepseek_8b_base": {
        "probe_acc": 0.38,
        "control_acc": 0.78,
        "category_multiplier": {
            "savyabhichara": 0.85,
            "viruddha": 1.1,
            "prakaranasama": 0.9,
            "sadhyasama": 1.0,
            "kalatita": 1.15,
        },
    },
    "stage0_pramana": {
        "probe_acc": 0.34,
        "control_acc": 0.70,
        "category_multiplier": {
            "savyabhichara": 0.95,
            "viruddha": 1.05,
            "prakaranasama": 0.85,
            "sadhyasama": 1.0,
            "kalatita": 1.1,
        },
    },
    "stage1_pramana": {
        "probe_acc": 0.52,
        "control_acc": 0.80,
        "category_multiplier": {
            "savyabhichara": 1.0,
            "viruddha": 1.05,
            "prakaranasama": 0.9,
            "sadhyasama": 1.0,
            "kalatita": 1.1,
        },
    },
    "base_with_cot": {
        "probe_acc": 0.44,
        "control_acc": 0.76,
        "category_multiplier": {
            "savyabhichara": 0.9,
            "viruddha": 1.1,
            "prakaranasama": 0.85,
            "sadhyasama": 1.0,
            "kalatita": 1.2,
        },
    },
    "base_with_nyaya_template": {
        "probe_acc": 0.42,
        "control_acc": 0.76,
        "category_multiplier": {
            "savyabhichara": 0.95,
            "viruddha": 1.05,
            "prakaranasama": 0.9,
            "sadhyasama": 1.0,
            "kalatita": 1.15,
        },
    },
}


def simulate_response(problem: dict, model_name: str, correct: bool) -> str:
    """Generate a synthetic model response."""
    if correct:
        if "pramana" in model_name:
            return (
                f"## Samshaya (Doubt Analysis)\n"
                f"The problem asks about {problem['problem_text'][:100]}...\n\n"
                f"## Pramana (Sources of Knowledge)\n"
                f"We use Anumana (inference) and Pratyaksha (direct observation).\n\n"
                f"## Pancha Avayava (5-Member Syllogism)\n"
                f"### Pratijna: The conclusion cannot be drawn from the premises.\n"
                f"### Hetu: There exists a counterexample that falsifies the universal.\n"
                f"### Udaharana: Wherever there is an exception to a pattern, the pattern cannot be universalized.\n"
                f"### Upanaya: In this case, a counterexample exists.\n"
                f"### Nigamana: Therefore, the conclusion does not follow.\n\n"
                f"## Tarka (Counterfactual Reasoning)\n"
                f"If the universal rule did hold, we would expect no counterexamples. But one exists.\n\n"
                f"## Hetvabhasa (Fallacy Check)\n"
                f"This is savyabhichara — the reason is erratic.\n\n"
                f"## Nirnaya (Ascertainment)\n"
                f"No, the answer is: the universal does not hold. Not all instances follow the pattern."
            )
        else:
            return (
                f"Let me analyze this step by step.\n\n"
                f"Looking at the premises, I need to check if the stated rule holds universally.\n\n"
                f"There is a counterexample that violates the proposed pattern. "
                f"Therefore, we cannot conclude the universal rule holds.\n\n"
                f"The answer is: No."
            )
    else:
        trap = problem.get("trap_answer", "Yes, based on the dominant pattern.")
        return (
            f"Based on the information given, the pattern suggests a clear conclusion.\n\n"
            f"The majority of cases follow this pattern, so we can reasonably conclude: {trap}\n\n"
            f"Yes."
        )


def generate_simulated_results(runner: VyaptiEvaluationRunner) -> dict[str, list[ProblemResult]]:
    """Generate simulated evaluation results based on expected accuracy profiles."""
    rng = random.Random(42)
    all_results = {}

    for model_name, profile in EXPECTED_PROFILES.items():
        print(f"\n=== Simulating: {model_name} ===")
        results = []

        for problem in runner.problems:
            is_probe = problem["type"] == "probe"
            base_acc = profile["probe_acc"] if is_probe else profile["control_acc"]
            cat_mult = profile["category_multiplier"].get(problem["category"], 1.0)
            effective_acc = min(1.0, base_acc * cat_mult)

            # Randomly determine if correct
            correct = rng.random() < effective_acc

            # Generate synthetic response
            response = simulate_response(problem, model_name, correct)

            # Build result
            solution = runner.solutions.get(problem["id"], {})
            result = ProblemResult(
                problem_id=problem["id"],
                category=problem["category"],
                problem_type=problem["type"],
                model_name=model_name,
                raw_response=response,
                response_length=len(response),
                generation_time_ms=rng.randint(500, 5000),
                final_answer_correct=correct,
            )

            # Tier 1: Answer correctness
            result.tiers.append(TierResult(
                tier=1, name="outcome", passed=correct,
                score=1.0 if correct else 0.0,
            ))

            # Tier 2: Structure (Pramana models use Nyaya format)
            has_structure = "pramana" in model_name and ("Samshaya" in response)
            result.tiers.append(TierResult(
                tier=2, name="structure", passed=has_structure,
                score=1.0 if has_structure else 0.0,
            ))

            # Tier 3: Vyapti
            vyapti_result = runner.vyapti_scorer.score(response, problem, solution)
            result.tiers.append(TierResult(
                tier=3, name="vyapti_explicitness", passed=vyapti_result.correct,
                score=1.0 if vyapti_result.correct else 0.0,
                details={"stated": vyapti_result.stated, "negation_detected": vyapti_result.negation_detected},
            ))

            # Tier 4: Z3 (simulated)
            result.tiers.append(TierResult(
                tier=4, name="z3_verification", passed=True,
                score=1.0,
                details={"simulated": True},
            ))

            # Tier 5: Hetvabhasa
            hclass = runner.hetvabhasa_classifier.classify(
                problem, response, solution, correct
            )
            result.hetvabhasa_classification = hclass.classified_type
            result.tiers.append(TierResult(
                tier=5, name="hetvabhasa_classification",
                passed=hclass.matches_ground_truth,
                score=hclass.confidence,
                details={
                    "classified_type": hclass.classified_type,
                    "ground_truth_type": hclass.ground_truth_type,
                },
            ))

            status = "CORRECT" if correct else "WRONG"
            print(f"  {problem['id']}: {status}")
            results.append(result)

        all_results[model_name] = results

    return all_results


# ═══════════════════════════════════════════════════════════════
# Real evaluation mode
# ═══════════════════════════════════════════════════════════════

def load_model_fn(model_config: dict, gen_config: dict):
    """Load a model and return a generate function."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        raise RuntimeError("torch/transformers not installed. Use --simulate for testing.")

    model_id = model_config["model_id"]
    print(f"  Loading {model_id}...")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Check for LoRA adapter
    try:
        from peft import AutoPeftModelForCausalLM
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map="auto",
            trust_remote_code=True,
        )
        print(f"  Loaded as PEFT model")
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map="auto",
            trust_remote_code=True,
        )
        print(f"  Loaded as base model")

    max_new_tokens = gen_config.get("max_new_tokens", 2048)
    temperature = gen_config.get("temperature", 0.5)
    top_p = gen_config.get("top_p", 0.75)
    top_k = gen_config.get("top_k", 5)

    def generate(prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=True,
            )
        response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
        return response

    return generate


def run_real_evaluation(config: dict, model_filter: str | None = None):
    """Run real evaluation with loaded models."""
    runner = VyaptiEvaluationRunner(config, project_root=PROJECT_ROOT)
    gen_config = config.get("generation", {})
    models_config = config.get("models", {})

    if model_filter:
        models_config = {k: v for k, v in models_config.items() if k == model_filter}

    model_fns = {}
    for name, mc in models_config.items():
        mc["prompt_style"] = mc.get("prompt_style", "direct")
        runner.config["prompt_style"] = mc["prompt_style"]
        print(f"\nLoading model: {name}")
        model_fns[name] = load_model_fn(mc, gen_config)

    results = runner.run_all(model_fns)

    # Save results
    output_dir = PROJECT_ROOT / config.get("results_path", "data/vyapti_probe/results")
    runner.save_results(results, output_dir)
    print(f"\nResults saved to {output_dir}")

    return results


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Run vyapti benchmark evaluation")
    parser.add_argument("--config", type=str, default="configs/vyapti_eval.yaml")
    parser.add_argument("--simulate", action="store_true", help="Generate simulated results")
    parser.add_argument("--model", type=str, default=None, help="Run single model only")
    args = parser.parse_args()

    if args.simulate:
        print("=== SIMULATION MODE ===")
        config = {
            "benchmark_path": "data/vyapti_probe/problems.json",
            "solutions_path": "data/vyapti_probe/solutions.json",
        }
        runner = VyaptiEvaluationRunner(config, project_root=PROJECT_ROOT)
        results = generate_simulated_results(runner)

        # Save
        output_dir = PROJECT_ROOT / "data" / "vyapti_probe" / "results"
        runner.save_results(results, output_dir)
        print(f"\nSimulated results saved to {output_dir}")
    else:
        import yaml
        config_path = PROJECT_ROOT / args.config
        with open(config_path) as f:
            config = yaml.safe_load(f)
        run_real_evaluation(config, args.model)


if __name__ == "__main__":
    main()
