#!/usr/bin/env python3
"""
Stage 0 Evaluation Script for Pramana.

Evaluates the fine-tuned Stage 0 model on held-out test examples by:
1. Loading the tuned model from models/stage_0/
2. Loading test examples (pramana-003 and pramana-005)
3. Generating model outputs with proper prompting
4. Running Tier 1 structural validation
5. Calculating format adherence metrics
6. Comparing outputs to ground truth
7. Saving results to results/stage_0_evaluation.json
"""

# IMPORTANT: Import unsloth first for optimizations
from unsloth import FastLanguageModel

import json
import logging
import os
import re
import yaml
from datetime import datetime
from pathlib import Path
from typing import Any

from pramana.application.data.parser import MarkdownParser, ParseError, ValidationError
from pramana.application.evaluation.handlers import Tier1StructuralHandler
from pramana.application.evaluation.pipeline import EvaluationPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Configuration
MODEL_DIR = os.getenv("MODEL_DIR", "models/stage_0")
SEED_EXAMPLES_DIR = "data/seed_examples/stage_zero"
RESULTS_DIR = "results"
RESULTS_FILE = os.getenv("RESULTS_FILE", "results/stage_0_evaluation.json")

# Format enforcement prompt (mirror training)
FORMAT_INSTRUCTIONS = """You MUST follow the exact markdown structure below. Do NOT add "Phase" labels or alternative headings. Use the exact headers and field labels shown.

Required section order:
1) ## Samshaya (Doubt Analysis)
2) ## Pramana (Sources of Knowledge)
3) ## Pancha Avayava (5-Member Syllogism)
4) ## Tarka (Counterfactual Reasoning)
5) ## Hetvabhasa (Fallacy Check)
6) ## Nirnaya (Ascertainment)

CRITICAL:
- Your response MUST start with: "## Samshaya (Doubt Analysis)"
- Copy the template exactly and fill in every field.
- Do not add any text before the first header or after the final field.

"""

FORMAT_TEMPLATE = """## Samshaya (Doubt Analysis)
**Doubt Type**:
**Justification**:

---

## Pramana (Sources of Knowledge)
### Pratyaksha (Direct Perception)
- 
### Anumana (Inference)
- 
### Upamana (Comparison)
- 
### Shabda (Testimony)
- 

---

## Pancha Avayava (5-Member Syllogism)
### Syllogism 1: 
**Pratijna (Thesis)**:
**Hetu (Reason)**:
**Udaharana (Universal + Example)**:
**Upanaya (Application)**:
**Nigamana (Conclusion)**:

---

## Tarka (Counterfactual Reasoning)
**Hypothesis**:
**Consequence**:
**Analysis**:
**Resolution**:

---

## Hetvabhasa (Fallacy Check)
Check for Savyabhichara: 
Check for Viruddha: 
Check for Asiddha: 
Check for Satpratipaksha: 
Check for Badhita: 

---

## Nirnaya (Ascertainment)
**Final Answer**:
**Justification**:
**Confidence**:
"""

SYSTEM_PROMPT = (
    "You are a Nyaya reasoning engine. Follow the exact output format provided."
)


def build_user_prompt(problem: str) -> str:
    """Build the user-visible prompt with format requirements."""
    return f"""### Problem:
{problem}

### Instructions:
{FORMAT_INSTRUCTIONS}

### Template:
{FORMAT_TEMPLATE}

### Nyaya Reasoning:
"""


def format_chat_text(
    tokenizer,
    user_prompt: str,
    assistant_response: str | None = None,
    add_generation_prompt: bool = False,
) -> str:
    """Format messages using the tokenizer chat template when available."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    if assistant_response is not None:
        messages.append({"role": "assistant", "content": assistant_response})

    if hasattr(tokenizer, "apply_chat_template") and getattr(
        tokenizer, "chat_template", None
    ):
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=add_generation_prompt
            )
        except Exception:
            pass

    # Fallback: plain concatenation
    if assistant_response is None:
        return user_prompt
    return f"{user_prompt}{assistant_response}"

# Test examples to evaluate (held-out from training)
TEST_EXAMPLE_IDS = ["pramana-003", "pramana-005"]

# Generation parameters
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.0
TOP_P = 1.0
TOP_K = 0
DO_SAMPLE = False


def extract_problem_from_markdown(content: str) -> str:
    """Extract problem statement from markdown content.
    
    Args:
        content: Markdown content with YAML frontmatter
        
    Returns:
        Problem statement text
    """
    # Remove YAML frontmatter
    pattern = r"^---\s*\n(.*?)^---\s*\n(.*)$"
    match = re.match(pattern, content, re.DOTALL | re.MULTILINE)
    if match:
        content_no_frontmatter = match.group(2)
    else:
        content_no_frontmatter = content
    
    # Extract problem section
    problem_pattern = r"^#\s+Problem\s*\n(.*?)(?=^##\s+|\Z)"
    problem_match = re.search(problem_pattern, content_no_frontmatter, re.MULTILINE | re.DOTALL)
    
    if not problem_match:
        raise ValueError("Missing '# Problem' section")
    
    return problem_match.group(1).strip()


def create_prompt(problem: str, tokenizer) -> str:
    """Create inference prompt in the same format as training.
    
    Args:
        problem: Problem statement text
        
    Returns:
        Formatted prompt string
    """
    user_prompt = build_user_prompt(problem)
    return format_chat_text(
        tokenizer=tokenizer,
        user_prompt=user_prompt,
        assistant_response=None,
        add_generation_prompt=True,
    )


def load_model(model_dir: str):
    """Load fine-tuned model from checkpoint directory.
    
    Args:
        model_dir: Path to model directory
        
    Returns:
        Tuple of (model, tokenizer)
        
    Raises:
        FileNotFoundError: If model directory doesn't exist
        RuntimeError: If model loading fails
    """
    model_path = Path(model_dir)
    if not model_path.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    logger.info(f"Loading model from {model_dir}...")
    
    try:
        # Check for HuggingFace token
        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
        
        # Load model with LoRA adapters
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(model_path.absolute()),
            max_seq_length=4096,
            dtype=None,  # Auto-detect
            load_in_4bit=True,  # Use 4-bit quantization
            token=hf_token,
        )
        
        # Enable inference mode
        FastLanguageModel.for_inference(model)
        
        logger.info("✓ Model loaded successfully")
        return model, tokenizer
        
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}") from e


def load_test_example(example_id: str, seed_dir: str) -> tuple[str, dict[str, Any]]:
    """Load a test example from seed examples directory.
    
    Args:
        example_id: Example ID (e.g., "pramana-003")
        seed_dir: Path to seed examples directory
        
    Returns:
        Tuple of (markdown content, parsed example dict)
        
    Raises:
        FileNotFoundError: If example file doesn't exist
    """
    seed_path = Path(seed_dir)
    
    # Find the markdown file matching the example ID
    pattern = f"{example_id}*.md"
    matching_files = list(seed_path.glob(pattern))
    
    if not matching_files:
        raise FileNotFoundError(f"No file found matching {pattern} in {seed_dir}")
    
    if len(matching_files) > 1:
        logger.warning(f"Multiple files match {pattern}, using {matching_files[0]}")
    
    file_path = matching_files[0]
    logger.info(f"Loading test example: {file_path.name}")
    
    markdown_content = file_path.read_text(encoding="utf-8")
    
    # Parse to get ground truth
    parser = MarkdownParser()
    try:
        parsed_example = parser.parse(markdown_content)
        # Convert to dict for JSON serialization
        example_dict = {
            "id": parsed_example.id,
            "problem_type": parsed_example.problem_type,
            "ground_truth": parsed_example.ground_truth,
            "difficulty": parsed_example.difficulty,
        }
    except (ParseError, ValidationError) as e:
        logger.warning(f"Failed to parse example for ground truth: {e}")
        example_dict = {"id": example_id}
    
    return markdown_content, example_dict


def generate_output(
    model, tokenizer, prompt: str, max_new_tokens: int = MAX_NEW_TOKENS
) -> str:
    """Generate model output for a given prompt.
    
    Args:
        model: Unsloth model instance (should be in inference mode)
        tokenizer: Tokenizer instance
        prompt: Input prompt
        max_new_tokens: Maximum number of tokens to generate
        
    Returns:
        Generated text
    """
    logger.info("Generating model output...")
    
    # Tokenize input
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=4096,
    )
    
    # Move to same device as model
    if hasattr(model, "device"):
        device = model.device
    else:
        import torch
        device = next(model.parameters()).device
    
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
        do_sample=DO_SAMPLE,
        pad_token_id=tokenizer.eos_token_id,
    )
    
    # Decode output
    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    
    logger.info(f"Generated {len(generated_text)} characters")
    return generated_text


def calculate_format_metrics(parsed_example) -> dict[str, Any]:
    """Calculate format adherence metrics from parsed example.
    
    Args:
        parsed_example: Parsed NyayaExample
        
    Returns:
        Dictionary with format metrics
    """
    metrics = {
        "phase_completeness": {
            "samshaya": parsed_example.samshaya is not None,
            "pramana": parsed_example.pramana is not None,
            "pancha_avayava": parsed_example.pancha_avayava is not None,
            "tarka": parsed_example.tarka is not None,
            "hetvabhasa": parsed_example.hetvabhasa is not None,
            "nirnaya": parsed_example.nirnaya is not None,
        },
        "num_phases_present": sum([
            parsed_example.samshaya is not None,
            parsed_example.pramana is not None,
            parsed_example.pancha_avayava is not None,
            parsed_example.tarka is not None,
            parsed_example.hetvabhasa is not None,
            parsed_example.nirnaya is not None,
        ]),
        "pramana_completeness": {
            "pratyaksha": len(parsed_example.pramana.pratyaksha) > 0,
            "anumana": len(parsed_example.pramana.anumana) > 0,
            "upamana": len(parsed_example.pramana.upamana) > 0,
            "shabda": len(parsed_example.pramana.shabda) > 0,
        },
        "num_pramana_sources": (
            len(parsed_example.pramana.pratyaksha)
            + len(parsed_example.pramana.anumana)
            + len(parsed_example.pramana.upamana)
            + len(parsed_example.pramana.shabda)
        ),
        "syllogism_completeness": {
            "num_syllogisms": len(parsed_example.pancha_avayava),
            "syllogisms_with_all_members": 0,
        },
    }
    
    # Check each syllogism for all 5 members
    for syllogism in parsed_example.pancha_avayava:
        members = [
            syllogism.pratijna,
            syllogism.hetu,
            syllogism.udaharana,
            syllogism.upanaya,
            syllogism.nigamana,
        ]
        if all(member and member.strip() for member in members):
            metrics["syllogism_completeness"]["syllogisms_with_all_members"] += 1
    
    return metrics


def evaluate_example(
    example_id: str,
    model,
    tokenizer,
    parser: MarkdownParser,
    evaluation_pipeline: EvaluationPipeline,
    seed_dir: str,
) -> dict[str, Any]:
    """Evaluate a single test example.
    
    Args:
        example_id: Example ID to evaluate
        model: Unsloth model instance
        tokenizer: Tokenizer instance
        parser: MarkdownParser instance
        evaluation_pipeline: EvaluationPipeline instance
        seed_dir: Path to seed examples directory
        
    Returns:
        Dictionary with evaluation results
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating: {example_id}")
    logger.info(f"{'='*60}")
    
    result = {
        "example_id": example_id,
        "timestamp": datetime.now().isoformat(),
    }
    
    try:
        # Load test example
        markdown_content, example_metadata = load_test_example(example_id, seed_dir)
        
        # Extract problem
        problem = extract_problem_from_markdown(markdown_content)
        result["problem"] = problem
        result["ground_truth"] = example_metadata.get("ground_truth", "")
        
        # Create prompt
        prompt = create_prompt(problem, tokenizer)
        result["prompt"] = prompt
        
        # Generate output
        generated_output = generate_output(model, tokenizer, prompt)
        result["generated_output"] = generated_output
        result["output_length"] = len(generated_output)
        
        # Try to parse generated output
        # The generated output should be the reasoning sections (Samshaya to Nirnaya)
        # We need to wrap it with minimal frontmatter for the parser
        # Extract problem from original markdown for context
        try:
            minimal_frontmatter = {
                "id": f"{example_id}-generated",
                "problem_type": example_metadata.get("problem_type", "unknown"),
                "ground_truth": example_metadata.get("ground_truth", ""),
            }
            frontmatter_yaml = yaml.dump(minimal_frontmatter, default_flow_style=False)
            
            # Normalize "Answer" to "Final Answer" for parser compatibility
            # The parser expects "Final Answer" but training data uses "Answer"
            normalized_output = re.sub(
                r"\*\*Answer\*\*:",
                "**Final Answer**:",
                generated_output,
                flags=re.IGNORECASE,
            )
            
            # Wrap generated output with frontmatter and problem section
            wrapped_output = f"""---
{frontmatter_yaml}---

# Problem

{problem}

{normalized_output}
"""
            
            parsed_output = parser.parse(wrapped_output)
            result["parse_success"] = True
            
            # Calculate format metrics
            format_metrics = calculate_format_metrics(parsed_output)
            result["format_metrics"] = format_metrics
            
            # Run evaluation pipeline
            pipeline_result = evaluation_pipeline.evaluate(parsed_output, generated_output)
            result["evaluation"] = {
                "overall_passed": pipeline_result.overall_passed,
                "final_tier": pipeline_result.final_tier,
                "total_duration_ms": pipeline_result.total_duration_ms,
                "tier_results": [
                    {
                        "tier": tr.tier,
                        "passed": tr.passed,
                        "score": tr.score,
                        "errors": tr.errors,
                        "details": tr.details,
                    }
                    for tr in pipeline_result.tier_results
                ],
            }
            
            # Compare to ground truth (simple string matching for now)
            ground_truth = example_metadata.get("ground_truth", "")
            if ground_truth:
                # Extract answer from Nirnaya section
                nirnaya_answer = parsed_output.nirnaya.answer if parsed_output.nirnaya else ""
                result["ground_truth_match"] = {
                    "ground_truth": ground_truth,
                    "model_answer": nirnaya_answer,
                    "exact_match": ground_truth.lower().strip() == nirnaya_answer.lower().strip(),
                }
            
        except (ParseError, ValidationError) as e:
            logger.warning(f"Failed to parse generated output: {e}")
            result["parse_success"] = False
            result["parse_error"] = str(e)
            result["evaluation"] = {
                "overall_passed": False,
                "final_tier": 0,
                "total_duration_ms": 0,
                "tier_results": [],
            }
        
        result["success"] = True
        
    except Exception as e:
        logger.error(f"Error evaluating {example_id}: {e}", exc_info=True)
        result["success"] = False
        result["error"] = str(e)
    
    return result


def print_evaluation_report(results: list[dict[str, Any]]) -> None:
    """Print detailed evaluation report to console.
    
    Args:
        results: List of evaluation result dictionaries
    """
    print("\n" + "=" * 80)
    print("STAGE 0 EVALUATION REPORT")
    print("=" * 80)
    
    for result in results:
        if not result.get("success"):
            print(f"\n❌ {result['example_id']}: FAILED")
            print(f"   Error: {result.get('error', 'Unknown error')}")
            continue
        
        print(f"\n{'='*80}")
        print(f"Example: {result['example_id']}")
        print(f"{'='*80}")
        
        # Parse status
        parse_success = result.get("parse_success", False)
        print(f"\nParse Status: {'✓ SUCCESS' if parse_success else '✗ FAILED'}")
        
        if not parse_success:
            print(f"   Parse Error: {result.get('parse_error', 'Unknown')}")
            continue
        
        # Format metrics
        format_metrics = result.get("format_metrics", {})
        phase_completeness = format_metrics.get("phase_completeness", {})
        num_phases = format_metrics.get("num_phases_present", 0)
        
        print(f"\nFormat Adherence:")
        print(f"  Phases Present: {num_phases}/6")
        print(f"    - Samshaya: {'✓' if phase_completeness.get('samshaya') else '✗'}")
        print(f"    - Pramana: {'✓' if phase_completeness.get('pramana') else '✗'}")
        print(f"    - Pancha Avayava: {'✓' if phase_completeness.get('pancha_avayava') else '✗'}")
        print(f"    - Tarka: {'✓' if phase_completeness.get('tarka') else '✗'}")
        print(f"    - Hetvabhasa: {'✓' if phase_completeness.get('hetvabhasa') else '✗'}")
        print(f"    - Nirnaya: {'✓' if phase_completeness.get('nirnaya') else '✗'}")
        
        # Pramana completeness
        pramana_completeness = format_metrics.get("pramana_completeness", {})
        num_pramana = format_metrics.get("num_pramana_sources", 0)
        print(f"\n  Pramana Sources: {num_pramana}")
        print(f"    - Pratyaksha: {'✓' if pramana_completeness.get('pratyaksha') else '✗'}")
        print(f"    - Anumana: {'✓' if pramana_completeness.get('anumana') else '✗'}")
        print(f"    - Upamana: {'✓' if pramana_completeness.get('upamana') else '✗'}")
        print(f"    - Shabda: {'✓' if pramana_completeness.get('shabda') else '✗'}")
        
        # Syllogism completeness
        syllogism_metrics = format_metrics.get("syllogism_completeness", {})
        num_syllogisms = syllogism_metrics.get("num_syllogisms", 0)
        complete_syllogisms = syllogism_metrics.get("syllogisms_with_all_members", 0)
        print(f"\n  Syllogisms: {num_syllogisms} total, {complete_syllogisms} complete")
        
        # Evaluation results
        evaluation = result.get("evaluation", {})
        overall_passed = evaluation.get("overall_passed", False)
        print(f"\nEvaluation Pipeline:")
        print(f"  Overall Status: {'✓ PASSED' if overall_passed else '✗ FAILED'}")
        print(f"  Final Tier: {evaluation.get('final_tier', 0)}")
        print(f"  Duration: {evaluation.get('total_duration_ms', 0)} ms")
        
        tier_results = evaluation.get("tier_results", [])
        for tr in tier_results:
            status = "✓ PASSED" if tr.get("passed") else "✗ FAILED"
            print(f"    Tier {tr.get('tier')}: {status} (score: {tr.get('score', 0.0):.2f})")
            if tr.get("errors"):
                for error in tr["errors"]:
                    print(f"      - {error}")
        
        # Ground truth comparison
        gt_match = result.get("ground_truth_match")
        if gt_match:
            print(f"\nGround Truth Comparison:")
            print(f"  Expected: {gt_match.get('ground_truth', '')}")
            print(f"  Model Answer: {gt_match.get('model_answer', '')}")
            exact_match = gt_match.get("exact_match", False)
            print(f"  Match: {'✓ EXACT' if exact_match else '✗ MISMATCH'}")
        
        # Output length
        print(f"\nOutput Statistics:")
        print(f"  Length: {result.get('output_length', 0)} characters")
    
    # Summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    successful = [r for r in results if r.get("success")]
    parseable = [r for r in successful if r.get("parse_success")]
    passed_eval = [r for r in parseable if r.get("evaluation", {}).get("overall_passed")]
    
    print(f"Total Examples: {len(results)}")
    print(f"Successful Runs: {len(successful)}")
    print(f"Parseable Outputs: {len(parseable)}/{len(successful)} ({100*len(parseable)/len(successful) if successful else 0:.1f}%)")
    print(f"Passed Evaluation: {len(passed_eval)}/{len(parseable)} ({100*len(passed_eval)/len(parseable) if parseable else 0:.1f}%)")
    
    if parseable:
        avg_phases = sum(
            r.get("format_metrics", {}).get("num_phases_present", 0) for r in parseable
        ) / len(parseable)
        print(f"Average Phases Present: {avg_phases:.1f}/6")
        
        avg_syllogisms = sum(
            r.get("format_metrics", {}).get("syllogism_completeness", {}).get("num_syllogisms", 0)
            for r in parseable
        ) / len(parseable)
        print(f"Average Syllogisms: {avg_syllogisms:.1f}")


def main() -> None:
    """Main evaluation function."""
    logger.info("=" * 60)
    logger.info("Pramana Stage 0 Evaluation")
    logger.info("=" * 60)
    
    # Create results directory
    results_path = Path(RESULTS_DIR)
    results_path.mkdir(parents=True, exist_ok=True)
    
    # Check model directory exists
    model_path = Path(MODEL_DIR)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model directory not found: {MODEL_DIR}\n"
            "Please train the model first using scripts/train_unsloth_dgx.py"
        )
    
    # Check seed examples directory exists
    seed_path = Path(SEED_EXAMPLES_DIR)
    if not seed_path.exists():
        raise FileNotFoundError(f"Seed examples directory not found: {SEED_EXAMPLES_DIR}")
    
    # Load model
    model, tokenizer = load_model(MODEL_DIR)
    
    # Initialize parser and evaluation pipeline
    parser = MarkdownParser()
    tier1_handler = Tier1StructuralHandler()
    evaluation_pipeline = EvaluationPipeline(handlers=[tier1_handler])
    
    # Evaluate each test example
    results = []
    for example_id in TEST_EXAMPLE_IDS:
        try:
            result = evaluate_example(
                example_id=example_id,
                model=model,
                tokenizer=tokenizer,
                parser=parser,
                evaluation_pipeline=evaluation_pipeline,
                seed_dir=SEED_EXAMPLES_DIR,
            )
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to evaluate {example_id}: {e}", exc_info=True)
            results.append({
                "example_id": example_id,
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            })
    
    # Save results
    results_file = Path(RESULTS_FILE)
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        "evaluation_timestamp": datetime.now().isoformat(),
        "model_dir": MODEL_DIR,
        "test_examples": TEST_EXAMPLE_IDS,
        "results": results,
    }
    
    with results_file.open("w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n✓ Results saved to {RESULTS_FILE}")
    
    # Print detailed report
    print_evaluation_report(results)
    
    logger.info("\n✓ Evaluation complete!")


if __name__ == "__main__":
    main()
