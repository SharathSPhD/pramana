"""Format adherence tests for Stage 0 model.

Following TDD principles:
1. Write test first (RED) - test MUST FAIL with current model
2. Implement fix (corrected training script)
3. Verify (GREEN) - test should PASS after retraining

This test validates that the model generates outputs with proper 6-phase Nyaya structure.
"""

import os
import re
from pathlib import Path
from typing import Any

import pytest

from pramana.application.data.parser import MarkdownParser, ParseError, ValidationError


# Test configuration
MODEL_DIR = os.getenv("MODEL_DIR", "models/stage_0")
SEED_EXAMPLES_DIR = "data/seed_examples/stage_zero"
TEST_EXAMPLE_IDS = ["pramana-003", "pramana-005"]
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

    if assistant_response is None:
        return user_prompt
    return f"{user_prompt}{assistant_response}"

# Generation parameters (matching evaluate_stage0.py)
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.0
TOP_P = 1.0
TOP_K = 0
DO_SAMPLE = False

# Expected 6 Nyaya phases
NYAYA_PHASES = [
    "samshaya",
    "pramana",
    "pancha_avayava",
    "tarka",
    "hetvabhasa",
    "nirnaya",
]


def parse_nyaya_phases(text: str) -> dict[str, bool]:
    """Check for all 6 phases in generated text.
    
    Args:
        text: Generated model output text
        
    Returns:
        Dictionary mapping phase names to boolean presence indicators
    """
    # Normalize text to lowercase for case-insensitive matching
    text_lower = text.lower()
    
    # Define patterns for each phase (flexible matching)
    phase_patterns = {
        "samshaya": [
            r"##\s*samshaya",
            r"samshaya\s*\(.*?doubt",
            r"doubt\s*analysis",
        ],
        "pramana": [
            r"##\s*pramana",
            r"pramana\s*\(.*?sources",
            r"sources\s*of\s*knowledge",
        ],
        "pancha_avayava": [
            r"##\s*pancha\s*avayava",
            r"pancha\s*avayava",
            r"5-member\s*syllogism",
            r"syllogism",
        ],
        "tarka": [
            r"##\s*tarka",
            r"tarka\s*\(.*?counterfactual",
            r"counterfactual\s*reasoning",
        ],
        "hetvabhasa": [
            r"##\s*hetvabhasa",
            r"hetvabhasa\s*\(.*?fallacy",
            r"fallacy\s*check",
        ],
        "nirnaya": [
            r"##\s*nirnaya",
            r"nirnaya\s*\(.*?ascertainment",
            r"ascertainment",
            r"final\s*answer",
        ],
    }
    
    phase_presence: dict[str, bool] = {}
    
    for phase_name, patterns in phase_patterns.items():
        # Check if any pattern matches
        found = any(re.search(pattern, text_lower, re.IGNORECASE) for pattern in patterns)
        phase_presence[phase_name] = found
    
    return phase_presence


def calculate_format_adherence(results: list[dict[str, Any]]) -> float:
    """Compute format adherence percentage across all test examples.
    
    Args:
        results: List of result dictionaries, each containing 'phase_presence' dict
        
    Returns:
        Format adherence percentage (0.0 to 100.0)
    """
    if not results:
        return 0.0
    
    total_phases_expected = len(NYAYA_PHASES) * len(results)
    total_phases_found = 0
    
    for result in results:
        phase_presence = result.get("phase_presence", {})
        phases_found = sum(1 for phase in NYAYA_PHASES if phase_presence.get(phase, False))
        total_phases_found += phases_found
    
    if total_phases_expected == 0:
        return 0.0
    
    adherence = (total_phases_found / total_phases_expected) * 100.0
    return adherence


def load_test_model(model_path: str):
    """Load model with Unsloth for inference.
    
    Args:
        model_path: Path to model directory
        
    Returns:
        Tuple of (model, tokenizer)
        
    Raises:
        FileNotFoundError: If model directory doesn't exist
        RuntimeError: If model loading fails
        ImportError: If unsloth is not installed
    """
    # Import unsloth first for optimizations
    try:
        from unsloth import FastLanguageModel
    except ImportError as e:
        raise ImportError(
            "Unsloth is not installed. Install with: pip install 'unsloth[colab-new]'"
        ) from e
    
    model_dir = Path(model_path)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_path}")
    
    # Check for HuggingFace token
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    
    # Load model with LoRA adapters
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(model_dir.absolute()),
        max_seq_length=4096,
        dtype=None,  # Auto-detect
        load_in_4bit=True,  # Use 4-bit quantization
        token=hf_token,
    )
    
    # Enable inference mode
    FastLanguageModel.for_inference(model)
    
    return model, tokenizer


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


def generate_output(
    model: Any, tokenizer: Any, prompt: str, max_new_tokens: int = MAX_NEW_TOKENS
) -> str:
    """Generate model output for a given prompt.
    
    Args:
        model: Unsloth model instance (should be in inference mode)
        tokenizer: Tokenizer instance
        prompt: Input prompt
        max_new_tokens: Maximum number of tokens to generate
        
    Returns:
        Generated text (only the generated part, without prompt)
    """
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
    
    return generated_text


def load_test_example(example_id: str, seed_dir: str) -> str:
    """Load a test example markdown file.
    
    Args:
        example_id: Example ID (e.g., "pramana-003")
        seed_dir: Path to seed examples directory
        
    Returns:
        Markdown content as string
        
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
        # Use first match if multiple found
        pass
    
    file_path = matching_files[0]
    return file_path.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def test_model():
    """Load test model once per test module.
    
    Yields:
        Tuple of (model, tokenizer)
        
    Note:
        This fixture is marked as module-scoped to avoid reloading the model
        for each test, which is expensive. The model is loaded once and reused.
    """
    project_root = Path(__file__).parent.parent
    model_path = project_root / MODEL_DIR
    
    if not model_path.exists():
        pytest.skip(f"Model directory not found: {model_path}. Train the model first.")
    
    try:
        model, tokenizer = load_test_model(str(model_path))
        yield model, tokenizer
    except ImportError as e:
        pytest.skip(f"Unsloth not available: {e}")
    except Exception as e:
        pytest.fail(f"Failed to load model: {e}")


@pytest.fixture
def parser():
    """Create a MarkdownParser instance."""
    return MarkdownParser()


@pytest.mark.slow
@pytest.mark.gpu
def test_stage0_format_adherence(test_model: tuple[Any, Any], parser: MarkdownParser) -> None:
    """Test that Stage 0 model generates outputs with proper 6-phase Nyaya structure.
    
    This test MUST FAIL with the current model (confirming the problem) and should
    PASS after retraining with the corrected script.
    
    Args:
        test_model: Fixture providing (model, tokenizer) tuple
        parser: MarkdownParser fixture
    """
    model, tokenizer = test_model
    
    project_root = Path(__file__).parent.parent
    seed_dir = project_root / SEED_EXAMPLES_DIR
    
    if not seed_dir.exists():
        pytest.fail(f"Seed examples directory not found: {seed_dir}")
    
    results: list[dict[str, Any]] = []
    
    # Test each example
    for example_id in TEST_EXAMPLE_IDS:
        try:
            # Load test example
            markdown_content = load_test_example(example_id, str(seed_dir))
            
            # Extract problem
            problem = extract_problem_from_markdown(markdown_content)
            
            # Create prompt
            prompt = create_prompt(problem, tokenizer)
            
            # Generate output
            generated_output = generate_output(model, tokenizer, prompt)
            
            # Parse phases from generated output
            phase_presence = parse_nyaya_phases(generated_output)
            
            # Try to parse with MarkdownParser for more detailed validation
            parse_success = False
            parse_error: str | None = None
            try:
                # Wrap generated output with minimal frontmatter for parser
                import yaml
                
                minimal_frontmatter = {
                    "id": f"{example_id}-generated",
                    "problem_type": "unknown",
                    "ground_truth": "unknown",
                }
                frontmatter_yaml = yaml.dump(minimal_frontmatter, default_flow_style=False)
                
                # Normalize "Answer" to "Final Answer" for parser compatibility
                normalized_output = re.sub(
                    r"\*\*Answer\*\*:",
                    "**Final Answer**:",
                    generated_output,
                    flags=re.IGNORECASE,
                )
                
                # Wrap generated output
                wrapped_output = f"""---
{frontmatter_yaml}---

# Problem

{problem}

{normalized_output}
"""
                
                parsed_output = parser.parse(wrapped_output)
                parse_success = True
                
                # Get more accurate phase presence from parsed output
                phase_presence = {
                    "samshaya": parsed_output.samshaya is not None,
                    "pramana": parsed_output.pramana is not None,
                    "pancha_avayava": len(parsed_output.pancha_avayava) > 0,
                    "tarka": parsed_output.tarka is not None,
                    "hetvabhasa": parsed_output.hetvabhasa is not None,
                    "nirnaya": parsed_output.nirnaya is not None,
                }
                
            except (ParseError, ValidationError) as e:
                parse_error = str(e)
                # Fall back to regex-based phase detection
            
            result = {
                "example_id": example_id,
                "phase_presence": phase_presence,
                "parse_success": parse_success,
                "parse_error": parse_error,
                "output_length": len(generated_output),
                "num_phases_found": sum(phase_presence.values()),
            }
            
            results.append(result)
            
        except Exception as e:
            pytest.fail(f"Failed to process {example_id}: {e}")
    
    # Calculate format adherence
    adherence = calculate_format_adherence(results)
    
    # Build detailed failure message
    failure_details: list[str] = []
    failure_details.append(f"\nFormat Adherence: {adherence:.1f}% (threshold: 80.0%)")
    failure_details.append(f"Test Examples: {len(results)}")
    
    for result in results:
        example_id = result["example_id"]
        phase_presence = result["phase_presence"]
        num_found = result["num_phases_found"]
        parse_success = result["parse_success"]
        
        failure_details.append(f"\n{example_id}:")
        failure_details.append(f"  Phases Found: {num_found}/6")
        failure_details.append(f"  Parse Success: {parse_success}")
        
        missing_phases = [
            phase for phase in NYAYA_PHASES if not phase_presence.get(phase, False)
        ]
        if missing_phases:
            failure_details.append(f"  Missing Phases: {', '.join(missing_phases)}")
        
        if result.get("parse_error"):
            failure_details.append(f"  Parse Error: {result['parse_error']}")
    
    failure_message = "\n".join(failure_details)
    
    # Assert format adherence > 80%
    # This MUST FAIL with current model
    assert (
        adherence > 80.0
    ), f"Format adherence {adherence:.1f}% is below threshold of 80.0%.{failure_message}"


@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.parametrize("example_id", TEST_EXAMPLE_IDS)
def test_individual_example_format(
    test_model: tuple[Any, Any],
    parser: MarkdownParser,
    example_id: str,
) -> None:
    """Test format adherence for individual examples.
    
    This provides more granular feedback about which specific examples fail.
    
    Args:
        test_model: Fixture providing (model, tokenizer) tuple
        parser: MarkdownParser fixture
        example_id: Example ID to test
    """
    model, tokenizer = test_model
    
    project_root = Path(__file__).parent.parent
    seed_dir = project_root / SEED_EXAMPLES_DIR
    
    # Load test example
    markdown_content = load_test_example(example_id, str(seed_dir))
    
    # Extract problem
    problem = extract_problem_from_markdown(markdown_content)
    
    # Create prompt
    prompt = create_prompt(problem, tokenizer)
    
    # Generate output
    generated_output = generate_output(model, tokenizer, prompt)
    
    # Parse phases
    phase_presence = parse_nyaya_phases(generated_output)
    
    # Check each required phase
    missing_phases = [
        phase for phase in NYAYA_PHASES if not phase_presence.get(phase, False)
    ]
    
    # Assert all phases are present
    assert (
        len(missing_phases) == 0
    ), f"Example {example_id} is missing phases: {', '.join(missing_phases)}. Found {sum(phase_presence.values())}/6 phases."
