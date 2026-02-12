#!/usr/bin/env python3
"""Real vyapti benchmark evaluation campaign.

Runs inside pramana-unsloth Docker container on DGX Spark (GB10, 120GB).
Loads models sequentially, evaluates all 100 problems per model, saves results.

Usage (inside Docker):
    cd /workspace/pramana
    python scripts/run_vyapti_real.py

Usage (from host):
    docker exec -i -e HF_TOKEN=... pramana-unsloth bash -c \
        "cd /workspace/pramana && python scripts/run_vyapti_real.py"
"""

import gc
import json
import os
import random
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pramana.benchmarks.vyapti_runner import (
    VyaptiEvaluationRunner,
    ProblemResult,
    TierResult,
)

HF_TOKEN = os.environ.get("HF_TOKEN", "")

# ═══════════════════════════════════════════════════════════════
# Model loading with memory management
# ═══════════════════════════════════════════════════════════════

_current_model = None
_current_tokenizer = None
_current_model_id = None


def resolve_git_hash(root: Path) -> str:
    """Return current git commit hash, if available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "not_available"


def unload_model():
    """Free GPU memory from current model."""
    global _current_model, _current_tokenizer, _current_model_id
    if _current_model is not None:
        print(f"  Unloading {_current_model_id}...")
        del _current_model
        del _current_tokenizer
        _current_model = None
        _current_tokenizer = None
        _current_model_id = None
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        mem_free = torch.cuda.mem_get_info()[0] / 1024**3
        print(f"  GPU memory free: {mem_free:.1f} GB")


def load_model(model_id: str):
    """Load a model, reusing if same model_id is already loaded."""
    global _current_model, _current_tokenizer, _current_model_id

    if _current_model_id == model_id:
        print(f"  Reusing already-loaded {model_id}")
        return _current_tokenizer, _current_model

    # Unload previous model first
    unload_model()

    print(f"  Loading {model_id}...")
    start = time.time()

    # Tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, token=HF_TOKEN, use_fast=True, trust_remote_code=True
        )
    except (ValueError, OSError) as exc:
        # Fallback for LlamaTokenizerFast issues
        if "TokenizersBackend" in str(exc) or "tokenizer" in str(exc).lower():
            from transformers import LlamaTokenizerFast
            tokenizer = LlamaTokenizerFast.from_pretrained(model_id, token=HF_TOKEN)
        else:
            raise

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Model -- use bfloat16, low_cpu_mem_usage (proven pattern from HF Space)
    device = torch.device("cuda")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=HF_TOKEN,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()

    elapsed = time.time() - start
    mem_used = torch.cuda.memory_allocated() / 1024**3
    print(f"  Loaded in {elapsed:.1f}s, GPU memory used: {mem_used:.1f} GB")

    _current_model = model
    _current_tokenizer = tokenizer
    _current_model_id = model_id

    return tokenizer, model


def make_generate_fn(model_id: str, prompt_style: str, gen_config: dict):
    """Create a generate function for a model configuration."""
    tokenizer, model = load_model(model_id)
    device = next(model.parameters()).device

    max_new_tokens = gen_config.get("max_new_tokens", 2048)
    temperature = gen_config.get("temperature", 0.5)
    top_p = gen_config.get("top_p", 0.75)
    top_k = gen_config.get("top_k", 5)
    seed = int(gen_config.get("seed", 42))
    generator = torch.Generator(device=device).manual_seed(seed)

    def generate(prompt: str) -> str:
        # Tokenize the prompt
        encoded = tokenizer(prompt, return_tensors="pt")
        input_ids = encoded.input_ids.to(device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=True,
                generator=generator,
                pad_token_id=tokenizer.pad_token_id,
            )
        response = tokenizer.decode(
            outputs[0][input_ids.shape[-1]:], skip_special_tokens=True
        )
        return response

    return generate


def _deserialize_problem_result(data: dict) -> ProblemResult:
    """Deserialize a per-problem JSON record into ProblemResult."""
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


# ═══════════════════════════════════════════════════════════════
# Main evaluation loop
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("VYAPTI PROBE BENCHMARK -- REAL EVALUATION CAMPAIGN")
    print("=" * 70)
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"HF_TOKEN: {'set' if HF_TOKEN else 'NOT SET'}")
    print()

    # Load config
    config_path = PROJECT_ROOT / "configs" / "vyapti_eval.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Initialize runner
    runner = VyaptiEvaluationRunner(config, project_root=PROJECT_ROOT)
    print(f"Loaded {len(runner.problems)} problems, {len(runner.solutions)} solutions")

    gen_config = config.get("generation", {})
    seed = int(gen_config.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"Seed: {seed}")

    models_config = config.get("models", {})
    output_dir = PROJECT_ROOT / config.get("results_path", "data/vyapti_probe/results_real")
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "timestamp": int(time.time()),
        "seed": seed,
        "git_hash": resolve_git_hash(PROJECT_ROOT),
        "generation": gen_config,
        "models": {
            model_name: {
                "model_id": mc.get("model_id", ""),
                "prompt_style": mc.get("prompt_style", "direct"),
            }
            for model_name, mc in models_config.items()
        },
    }
    with open(output_dir / "run_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    all_results = {}
    total_start = time.time()

    for model_name, mc in models_config.items():
        model_id = mc["model_id"]
        prompt_style = mc.get("prompt_style", "direct")

        model_dir = output_dir / model_name
        summary_file = model_dir / "summary.json"
        model_dir.mkdir(parents=True, exist_ok=True)
        existing_results: dict[str, ProblemResult] = {}
        for rf in sorted(model_dir.glob("*.json")):
            if rf.name == "summary.json":
                continue
            with open(rf) as f:
                result_data = json.load(f)
            if "problem_id" in result_data:
                existing_results[result_data["problem_id"]] = _deserialize_problem_result(result_data)

        remaining = [p for p in runner.problems if p["id"] not in existing_results]
        if not remaining:
            print(f"\n=== SKIPPING {model_name} (all problems already completed) ===")
        else:
            print(f"\n{'=' * 70}")
            print(f"EVALUATING: {model_name}")
            print(f"  Model: {model_id}")
            print(f"  Prompt: {prompt_style}")
            print(f"  Checkpointed: {len(existing_results)} done, {len(remaining)} remaining")
            print(f"{'=' * 70}")

            # Set prompt style on runner
            runner.config["prompt_style"] = prompt_style

            # Create generate function (loads model if needed)
            try:
                gen_fn = make_generate_fn(model_id, prompt_style, gen_config)
            except Exception as e:
                print(f"  ERROR loading model: {e}")
                continue

            model_start = time.time()
            for idx, problem in enumerate(remaining, start=1):
                print(f"  [{idx}/{len(remaining)}] {problem['id']}...", end=" ", flush=True)
                result = runner.evaluate_problem(model_name, gen_fn, problem)
                print(f"{'CORRECT' if result.final_answer_correct else 'WRONG'}")
                existing_results[result.problem_id] = result
                with open(model_dir / f"{result.problem_id}.json", "w") as f:
                    json.dump(asdict(result), f, indent=2, default=str)

            model_elapsed = time.time() - model_start
            print(f"\n  Completed remaining {len(remaining)} problems in {model_elapsed:.0f}s ({model_elapsed/60:.1f}min)")

        ordered_results = [
            existing_results[p["id"]]
            for p in runner.problems
            if p["id"] in existing_results
        ]
        if len(ordered_results) != len(runner.problems):
            print(f"  ERROR: only {len(ordered_results)}/{len(runner.problems)} problems available for {model_name}")
            continue

        # Save model summary
        summary = runner._compute_summary(model_name, ordered_results)
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        # Print summary
        print(f"\n  --- {model_name} Summary ---")
        print(f"  Accuracy: {summary['correct']}/{summary['total']} ({summary['accuracy']:.1%})")
        print(f"  Probe:    {summary['probe_accuracy']:.1%}")
        print(f"  Control:  {summary['control_accuracy']:.1%}")
        print(f"  Gap:      {summary['control_accuracy'] - summary['probe_accuracy']:.1%}")

        all_results[model_name] = [asdict(r) for r in ordered_results]

    # Save cross-model summary
    cross_summary = {}
    for model_name in models_config:
        model_dir = output_dir / model_name
        sf = model_dir / "summary.json"
        if sf.exists():
            with open(sf) as f:
                cross_summary[model_name] = json.load(f)

    with open(output_dir / "summary.json", "w") as f:
        json.dump(cross_summary, f, indent=2)

    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 70}")
    print(f"ALL EVALUATIONS COMPLETE in {total_elapsed:.0f}s ({total_elapsed/3600:.1f}h)")
    print(f"Results saved to {output_dir}")
    print(f"{'=' * 70}")

    # Print final leaderboard
    print(f"\n{'Model':<28} {'Acc':>6} {'Probe':>6} {'Ctrl':>6} {'Gap':>6}")
    print("-" * 56)
    for name, data in cross_summary.items():
        gap = data["control_accuracy"] - data["probe_accuracy"]
        print(f"{name:<28} {data['accuracy']:>5.1%} {data['probe_accuracy']:>5.1%} "
              f"{data['control_accuracy']:>5.1%} {gap:>+5.1%}")

    # Clean up
    unload_model()


if __name__ == "__main__":
    main()
