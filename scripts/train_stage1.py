#!/usr/bin/env python3
"""
Stage 1 Training Script with Unsloth on DGX Spark.
"""

# IMPORTANT: Import unsloth first for optimizations
from unsloth import FastLanguageModel, FastModel

import json
import os
import re
import math
from pathlib import Path

from datasets import Dataset
from trl import SFTTrainer, SFTConfig
import torch

from pramana.application.training.callbacks import NyayaMetricsCallback

# Configuration
MAX_SEQ_LENGTH = int(os.getenv("MAX_SEQ_LENGTH", "4096"))
NUM_TRAIN_EPOCHS = int(os.getenv("NUM_TRAIN_EPOCHS", "10"))
LORA_RANK = int(os.getenv("LORA_RANK", "64"))
LORA_ALPHA = int(os.getenv("LORA_ALPHA", str(LORA_RANK)))
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "models/stage_1")
MODEL_NAME = os.getenv(
    "MODEL_NAME", "unsloth/DeepSeek-R1-Distill-Llama-8B-bnb-4bit"
)
DATA_DIRS = [
    "data/seed_examples/stage_zero",
    "data/seed_examples/stage_one",
]

# Create output directory
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


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

SYSTEM_PROMPT = "You are a Nyaya reasoning engine. Follow the exact output format provided."


def remove_frontmatter(content: str) -> str:
    pattern = r"^---\s*\n(.*?)^---\s*\n(.*)$"
    match = re.match(pattern, content, re.DOTALL | re.MULTILINE)
    if match:
        return match.group(2)
    return content


def extract_problem(content: str) -> str:
    pattern = r"^#\s+Problem\s*\n(.*?)(?=^##\s+|\Z)"
    match = re.search(pattern, content, re.MULTILINE | re.DOTALL)
    if not match:
        raise ValueError("Missing '# Problem' section")
    return match.group(1).strip()


def extract_reasoning_trace(content: str) -> str:
    samshaya_start = re.search(r"^##\s+Samshaya", content, re.MULTILINE)
    if not samshaya_start:
        raise ValueError("Missing '## Samshaya' section")
    reasoning = content[samshaya_start.start():].strip()
    if "## Nirnaya" not in reasoning:
        raise ValueError("Missing '## Nirnaya' section")
    return reasoning


def build_user_prompt(problem: str) -> str:
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


def load_training_data(tokenizer) -> Dataset:
    examples = []
    for data_dir in DATA_DIRS:
        for md_file in sorted(Path(data_dir).glob("*.md")):
            if md_file.name.lower() == "readme.md":
                continue
            content = md_file.read_text(encoding="utf-8")
            content_no_frontmatter = remove_frontmatter(content)
            problem = extract_problem(content_no_frontmatter)
            reasoning = extract_reasoning_trace(content_no_frontmatter)
            user_prompt = build_user_prompt(problem)
            text = format_chat_text(
                tokenizer,
                user_prompt,
                assistant_response=reasoning,
                add_generation_prompt=False,
            )
            examples.append({"text": text})

    if not examples:
        raise ValueError("No training examples found.")

    print(f"Loaded {len(examples)} training examples")
    return Dataset.from_list(examples)


def main() -> None:
    print("=" * 60)
    print("Pramana Stage 1 Training with Unsloth on DGX Spark")
    print("=" * 60)

    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if hf_token:
        print(f"✓ HF_TOKEN found (length: {len(hf_token)})")
    else:
        print("⚠ No HF_TOKEN found - using non-gated models only")

    print("\n[1/5] Loading model with Unsloth...")
    model, tokenizer = FastModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        load_in_8bit=False,
        full_finetuning=False,
        token=hf_token,
    )

    print("\n[2/5] Applying LoRA...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=LORA_ALPHA,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
        max_seq_length=MAX_SEQ_LENGTH,
        use_rslora=False,
        loftq_config=None,
    )

    model.print_trainable_parameters()

    print("\n[3/5] Loading training data...")
    dataset = load_training_data(tokenizer)

    print("\n[4/5] Creating train/validation split (80/20)...")
    split_dataset = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    print(f"Training examples: {len(train_dataset)}")
    print(f"Validation examples: {len(eval_dataset)}")

    # Build validation prompt for metrics callback
    sample_text = train_dataset[0]["text"]
    if "### Problem:" in sample_text and "### Instructions:" in sample_text:
        sample_problem = sample_text.split("### Problem:")[1].split(
            "### Instructions:"
        )[0].strip()
    else:
        sample_problem = "Solve the problem using Nyaya structure."

    validation_prompt = build_user_prompt(sample_problem)
    validation_prompt = format_chat_text(
        tokenizer, validation_prompt, assistant_response=None, add_generation_prompt=True
    )

    nyaya_callback = NyayaMetricsCallback(
        tokenizer=tokenizer,
        prompt=validation_prompt,
        max_new_tokens=512,
    )

    # Calculate steps per epoch
    per_device_batch_size = int(os.getenv("BATCH_SIZE", "1"))
    gradient_accumulation_steps = int(os.getenv("GRAD_ACCUM", "4"))
    effective_batch_size = per_device_batch_size * gradient_accumulation_steps
    steps_per_epoch = max(1, math.ceil(len(train_dataset) / effective_batch_size))
    max_steps = steps_per_epoch * NUM_TRAIN_EPOCHS

    print("\n[5/5] Setting up SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        args=SFTConfig(
            max_seq_length=MAX_SEQ_LENGTH,
            per_device_train_batch_size=per_device_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=4,
            max_steps=max_steps,
            num_train_epochs=NUM_TRAIN_EPOCHS,
            eval_strategy="steps",
            eval_steps=max(1, steps_per_epoch),
            save_strategy="steps",
            save_steps=max(1, steps_per_epoch),
            logging_steps=max(1, steps_per_epoch // 2),
            output_dir=OUTPUT_DIR,
            optim="adamw_8bit",
            learning_rate=2e-5,
            seed=42,
            fp16=False,
            bf16=True,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        ),
        dataset_text_field="text",
        callbacks=[nyaya_callback],
    )

    print("\n" + "=" * 60)
    print("Starting Stage 1 training...")
    print("=" * 60)
    trainer.train()

    print("\n" + "=" * 60)
    print("Saving model...")
    print("=" * 60)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"\n✓ Model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
