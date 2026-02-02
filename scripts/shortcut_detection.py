#!/usr/bin/env python3
"""Shortcut detection ablation study for Stage 0 model."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

from pramana.application.data.parser import MarkdownParser, ParseError, ValidationError
from pramana.application.evaluation.ablation import (
    build_baseline_prompt,
    build_nyaya_prompt,
    extract_answer_from_output,
)
from pramana.application.evaluation.scoring import score_answers
from pramana.application.evaluation.model_loader import should_use_unsloth


MODEL_DIR = os.getenv("MODEL_DIR", "models/stage_0_corrected")
BASE_MODEL_NAME = os.getenv("BASE_MODEL_NAME", "unsloth/Llama-3.2-3B-Instruct-bnb-4bit")
BASE_MODEL_NAME_CPU = os.getenv("BASE_MODEL_NAME_CPU", "unsloth/Llama-3.2-3B-Instruct")
VALIDATION_DIR = os.getenv("VALIDATION_DIR", "data/validation/stage_zero")
RESULTS_FILE = os.getenv("RESULTS_FILE", "results/stage_0_shortcut_detection.json")

MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "256"))
TEMPERATURE = 0.0
TOP_P = 1.0
TOP_K = 0
DO_SAMPLE = False
SEMANTIC_THRESHOLD = float(os.getenv("SEMANTIC_THRESHOLD", "0.7"))

SYSTEM_PROMPT_NYAYA = "You are a Nyaya reasoning engine. Follow the exact output format provided."
SYSTEM_PROMPT_BASELINE = "You are a careful reasoning assistant."

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


def format_chat_text(tokenizer, system_prompt: str, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    return user_prompt


def load_model(model_name_or_dir: str):
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
                model_name=model_name_or_dir,
                max_seq_length=4096,
                dtype=None,
                load_in_4bit=True,
                token=hf_token,
            )
            FastLanguageModel.for_inference(model)
            return model, tokenizer
        except Exception:
            pass

    # CPU / transformers fallback
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = Path(model_name_or_dir)
    use_adapter = model_path.exists() and (model_path / "adapter_config.json").exists()

    if use_adapter:
        from peft import PeftModel

        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME_CPU, device_map="cpu", torch_dtype="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME_CPU)
        model = PeftModel.from_pretrained(base_model, model_name_or_dir)
    else:
        resolved_name = (
            BASE_MODEL_NAME_CPU
            if model_name_or_dir == BASE_MODEL_NAME
            else model_name_or_dir
        )
        model = AutoModelForCausalLM.from_pretrained(
            resolved_name, device_map="cpu", torch_dtype="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(resolved_name)

    model.eval()
    return model, tokenizer


def generate_output(model, tokenizer, prompt: str) -> str:
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=4096,
    )
    device = model.device if hasattr(model, "device") else next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
        do_sample=DO_SAMPLE,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


def load_validation_examples() -> list[dict[str, str]]:
    parser = MarkdownParser()
    examples = []
    for path in sorted(Path(VALIDATION_DIR).glob("*.md")):
        content = path.read_text(encoding="utf-8")
        try:
            parsed = parser.parse(content)
            examples.append(
                {
                    "id": parsed.id,
                    "problem": parsed.problem,
                    "ground_truth": parsed.ground_truth,
                }
            )
        except (ParseError, ValidationError):
            continue
    limit = int(os.getenv("MAX_EXAMPLES", "0"))
    return examples[:limit] if limit > 0 else examples


def main() -> None:
    results_path = Path(RESULTS_FILE)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    examples = load_validation_examples()
    if not examples:
        raise RuntimeError(f"No validation examples found in {VALIDATION_DIR}")

    tuned_model, tuned_tokenizer = load_model(MODEL_DIR)
    base_model, base_tokenizer = load_model(BASE_MODEL_NAME)

    conditions = {
        "nyaya_tuned_with_instructions": {
            "model": tuned_model,
            "tokenizer": tuned_tokenizer,
            "system_prompt": SYSTEM_PROMPT_NYAYA,
            "prompt_builder": lambda problem: build_nyaya_prompt(
                problem=problem,
                format_instructions=FORMAT_INSTRUCTIONS,
                format_template=FORMAT_TEMPLATE,
            ),
        },
        "nyaya_tuned_without_instructions": {
            "model": tuned_model,
            "tokenizer": tuned_tokenizer,
            "system_prompt": SYSTEM_PROMPT_BASELINE,
            "prompt_builder": build_baseline_prompt,
        },
        "base_model_with_instructions": {
            "model": base_model,
            "tokenizer": base_tokenizer,
            "system_prompt": SYSTEM_PROMPT_NYAYA,
            "prompt_builder": lambda problem: build_nyaya_prompt(
                problem=problem,
                format_instructions=FORMAT_INSTRUCTIONS,
                format_template=FORMAT_TEMPLATE,
            ),
        },
    }

    output = {
        "timestamp": datetime.now().isoformat(),
        "validation_dir": VALIDATION_DIR,
        "model_dir": MODEL_DIR,
        "base_model": BASE_MODEL_NAME,
        "results": [],
        "summary": {},
    }

    for idx, example in enumerate(examples, start=1):
        example_result = {
            "example_id": example["id"],
            "ground_truth": example["ground_truth"],
            "conditions": {},
        }
        for condition_name, config in conditions.items():
            print(f"[{idx}/{len(examples)}] Running {condition_name}...")
            prompt = config["prompt_builder"](example["problem"])
            formatted_prompt = format_chat_text(
                config["tokenizer"], config["system_prompt"], prompt
            )
            generated = generate_output(config["model"], config["tokenizer"], formatted_prompt)
            answer = extract_answer_from_output(generated)
            scores = score_answers(
                predicted=answer,
                ground_truth=example["ground_truth"],
                threshold=SEMANTIC_THRESHOLD,
                use_embeddings=True,
            )
            example_result["conditions"][condition_name] = {
                "answer": answer,
                "scores": scores,
            }
        output["results"].append(example_result)

    # Summaries
    for condition_name in conditions:
        matches = [
            r["conditions"][condition_name]["scores"]["semantic_match"]
            for r in output["results"]
        ]
        accuracy = sum(1 for m in matches if m) / len(matches) if matches else 0.0
        output["summary"][condition_name] = {
            "semantic_accuracy": accuracy,
            "total_examples": len(matches),
        }

    with results_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("Shortcut detection results saved to", RESULTS_FILE)


if __name__ == "__main__":
    main()
