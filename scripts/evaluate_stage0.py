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
from pramana.application.evaluation.llm_judge import Tier2LLMJudgeHandler
from pramana.application.evaluation.pipeline import EvaluationPipeline
from pramana.application.evaluation.content_quality import ContentQualityValidator
from pramana.application.evaluation.scoring import score_answers, wilson_interval
from pramana.application.evaluation.model_loader import should_use_unsloth
from pramana.application.evaluation.z3_handler import Tier3Z3VerifierHandler
from pramana.config.settings import PramanaSettings
from pramana.infrastructure.llm import LLMClientError, create_llm_client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Configuration
MODEL_DIR = os.getenv("MODEL_DIR", "models/stage_0")
VALIDATION_DIR = os.getenv("VALIDATION_DIR", "data/validation/stage_zero")
BASE_MODEL_NAME_CPU = os.getenv("BASE_MODEL_NAME_CPU", "unsloth/Llama-3.2-3B-Instruct")
EVAL_TIERS = os.getenv("EVAL_TIERS", "1")
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
TEST_EXAMPLE_IDS_ENV = os.getenv("TEST_EXAMPLE_IDS")

# Generation parameters
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.0
TOP_P = 1.0
TOP_K = 0
DO_SAMPLE = False
SEMANTIC_THRESHOLD = float(os.getenv("SEMANTIC_THRESHOLD", "0.7"))


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
    
    # Check for HuggingFace token
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    prefer_unsloth = os.getenv("USE_UNSLOTH", "1") == "1"

    torch_available = False
    has_gpu = False
    try:
        import torch

        torch_available = True
        has_gpu = torch.cuda.is_available()
    except Exception:
        pass

    if should_use_unsloth(
        prefer_unsloth=prefer_unsloth, torch_available=torch_available, has_gpu=has_gpu
    ):
        try:
            from unsloth import FastLanguageModel

            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=str(model_path.absolute()),
                max_seq_length=4096,
                dtype=None,  # Auto-detect
                load_in_4bit=True,  # Use 4-bit quantization
                token=hf_token,
            )
            FastLanguageModel.for_inference(model)
            logger.info("✓ Model loaded successfully (Unsloth)")
            return model, tokenizer
        except Exception:
            pass

    # CPU / transformers fallback
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME_CPU, device_map="cpu", torch_dtype="auto", token=hf_token
        )
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME_CPU, token=hf_token)
        model = PeftModel.from_pretrained(base_model, str(model_path.absolute()))
        model.eval()
        logger.info("✓ Model loaded successfully (Transformers + PEFT)")
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


def list_validation_ids(seed_dir: str) -> list[str]:
    """Derive example IDs from validation directory filenames."""
    ids: list[str] = []
    for path in sorted(Path(seed_dir).glob("*.md")):
        stem_parts = path.stem.split("-")
        if len(stem_parts) >= 2:
            ids.append("-".join(stem_parts[:2]))
        else:
            ids.append(path.stem)
    return ids


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

            # Content quality metrics
            content_validator = ContentQualityValidator()
            content_quality = content_validator.validate(parsed_output)
            result["content_quality"] = {
                "pratyaksha_score": content_quality.pratyaksha_score,
                "udaharana_valid": content_quality.udaharana_valid,
                "tarka_meaningful": content_quality.tarka_meaningful,
                "hetvabhasa_completeness": content_quality.hetvabhasa_completeness,
                "overall_score": content_quality.overall_score,
            }
            
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
            
            # Compare to ground truth with semantic scoring
            ground_truth = example_metadata.get("ground_truth", "")
            if ground_truth:
                # Extract answer from Nirnaya section
                nirnaya_answer = parsed_output.nirnaya.answer if parsed_output.nirnaya else ""
                answer_scores = score_answers(
                    predicted=nirnaya_answer,
                    ground_truth=ground_truth,
                    threshold=SEMANTIC_THRESHOLD,
                    use_embeddings=True,
                )
                result["ground_truth_match"] = {
                    "ground_truth": ground_truth,
                    "model_answer": nirnaya_answer,
                    **answer_scores,
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

        semantic_matches = [
            r.get("ground_truth_match", {}).get("semantic_match")
            for r in parseable
            if r.get("ground_truth_match") is not None
        ]
        semantic_rate = (
            sum(1 for m in semantic_matches if m) / len(semantic_matches)
            if semantic_matches
            else 0.0
        )
        print(f"Semantic Match Rate: {semantic_rate:.2%}")


def parse_eval_tiers(value: str) -> list[int]:
    """Parse evaluation tiers from environment."""
    if not value:
        return [1]
    if value.strip().lower() == "all":
        return [1, 2, 3]
    tiers = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            tier_num = int(item)
        except ValueError:
            continue
        if tier_num in (1, 2, 3):
            tiers.append(tier_num)
    return tiers or [1]


def pearson_correlation(xs: list[float], ys: list[float]) -> float | None:
    """Compute Pearson correlation; returns None if undefined."""
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    denom = (var_x * var_y) ** 0.5
    if denom == 0:
        return None
    return cov / denom


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
    seed_path = Path(VALIDATION_DIR)
    if not seed_path.exists():
        raise FileNotFoundError(f"Validation directory not found: {VALIDATION_DIR}")
    
    # Load model
    model, tokenizer = load_model(MODEL_DIR)
    
    # Initialize parser and evaluation pipeline
    parser = MarkdownParser()
    tiers_to_run = parse_eval_tiers(EVAL_TIERS)
    handlers = []
    if 1 in tiers_to_run:
        handlers.append(Tier1StructuralHandler())
    if 2 in tiers_to_run:
        try:
            llm_client = create_llm_client(PramanaSettings())
            handlers.append(Tier2LLMJudgeHandler(llm_client=llm_client))
        except LLMClientError as exc:
            logger.warning("Tier 2 judge unavailable: %s", exc)
    if 3 in tiers_to_run:
        handlers.append(Tier3Z3VerifierHandler())
    if not handlers:
        raise RuntimeError("No valid evaluation tiers configured.")
    evaluation_pipeline = EvaluationPipeline(handlers=handlers)
    
    # Determine validation examples
    if TEST_EXAMPLE_IDS_ENV:
        test_example_ids = [
            example_id.strip()
            for example_id in TEST_EXAMPLE_IDS_ENV.split(",")
            if example_id.strip()
        ]
    else:
        test_example_ids = list_validation_ids(VALIDATION_DIR)

    # Evaluate each test example
    results = []
    for example_id in test_example_ids:
        try:
            result = evaluate_example(
                example_id=example_id,
                model=model,
                tokenizer=tokenizer,
                parser=parser,
                evaluation_pipeline=evaluation_pipeline,
                seed_dir=VALIDATION_DIR,
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
        "test_examples": test_example_ids,
        "results": results,
    }

    # Summary metrics with confidence intervals
    parseable = [r for r in results if r.get("parse_success")]
    format_full = [
        r
        for r in parseable
        if r.get("format_metrics", {}).get("num_phases_present") == 6
    ]
    format_rate = len(format_full) / len(results) if results else 0.0
    format_ci = wilson_interval(successes=len(format_full), total=len(results))

    semantic_matches = [
        r.get("ground_truth_match", {}).get("semantic_match")
        for r in parseable
        if r.get("ground_truth_match") is not None
    ]
    semantic_successes = sum(1 for m in semantic_matches if m)
    semantic_total = len(semantic_matches)
    semantic_rate = semantic_successes / semantic_total if semantic_total else 0.0
    semantic_ci = wilson_interval(successes=semantic_successes, total=semantic_total)

    output_data["summary"] = {
        "format_adherence": {
            "rate": format_rate,
            "confidence_interval_95": list(format_ci),
        },
        "answer_correctness": {
            "semantic_match": semantic_rate,
            "confidence_interval_95": list(semantic_ci),
        },
    }

    # Structure-accuracy correlation (phase completeness vs semantic similarity)
    structure_scores: list[float] = []
    accuracy_scores: list[float] = []
    for result in results:
        format_metrics = result.get("format_metrics", {})
        ground_truth_match = result.get("ground_truth_match", {})
        if not format_metrics or not ground_truth_match:
            continue
        if ground_truth_match.get("semantic_similarity") is None:
            continue
        structure_scores.append(format_metrics.get("num_phases_present", 0) / 6)
        accuracy_scores.append(ground_truth_match.get("semantic_similarity", 0.0))

    output_data["summary"]["structure_accuracy_correlation"] = pearson_correlation(
        structure_scores, accuracy_scores
    )
    
    with results_file.open("w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n✓ Results saved to {RESULTS_FILE}")
    
    # Print detailed report
    print_evaluation_report(results)
    
    logger.info("\n✓ Evaluation complete!")


if __name__ == "__main__":
    main()
