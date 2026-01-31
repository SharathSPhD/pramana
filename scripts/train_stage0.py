#!/usr/bin/env python3
"""Stage 0 proof-of-concept training script.

This script trains Llama-3.2-3B-Instruct on 5 seed examples using LoRA fine-tuning.
Uses standard HuggingFace transformers (not unsloth) for compatibility.
"""

import json
import os
from pathlib import Path

from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer

# Configuration
# Using TinyLlama for proof-of-concept (non-gated, small model)
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OUTPUT_DIR = "models/stage_0"
DATA_PATH = "data/training/stage_0.jsonl"

# Ensure output directory exists
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


def load_data() -> Dataset:
    """Load training data from JSONL file.

    Returns:
        Dataset with formatted examples
    """
    examples = []
    data_path = Path(DATA_PATH)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found at {DATA_PATH}")
    
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))
    
    if not examples:
        raise ValueError(f"No examples found in {DATA_PATH}")
    
    print(f"Loaded {len(examples)} training examples")
    return Dataset.from_list(examples)


def format_prompt(example: dict) -> str:
    """Format example as instruction-response prompt.

    Args:
        example: Dictionary with 'instruction' and 'output' fields

    Returns:
        Formatted prompt string
    """
    instruction = example.get("instruction", "")
    output = example.get("output", "")
    
    # Format as instruction-following prompt
    prompt = f"""### Problem:
{instruction}

### Nyaya Reasoning:
{output}"""
    
    return prompt


def main() -> None:
    """Main training function."""
    # Check for HuggingFace token
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if not hf_token:
        print("Warning: HF_TOKEN or HUGGINGFACE_HUB_TOKEN not set. "
              "May need authentication for model download.")
    
    print("=" * 60)
    print("Stage 0 Training Script")
    print("=" * 60)
    
    # Load tokenizer
    print("\n[1/6] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        token=hf_token,
        trust_remote_code=True,
    )
    
    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model
    print("\n[2/6] Loading model...")
    # Note: GB10 (sm_121) not yet supported by PyTorch, using CPU for proof-of-concept
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        token=hf_token,
        torch_dtype="float32",  # Use float32 for CPU
        trust_remote_code=True,
    )
    
    # Apply LoRA
    print("\n[3/6] Applying LoRA...")
    lora_config = LoraConfig(
        r=32,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    # Load data
    print("\n[4/6] Loading training data...")
    dataset = load_data()
    
    # Format dataset
    print("\n[5/6] Formatting dataset...")
    formatted_dataset = dataset.map(
        lambda x: {"text": format_prompt(x)},
        remove_columns=dataset.column_names,
    )
    
    # Setup training arguments
    print("\n[6/6] Setting up trainer...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=2,  # Reduced for CPU proof-of-concept
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        learning_rate=2e-5,
        warmup_steps=2,  # Use warmup_steps instead of deprecated warmup_ratio
        logging_steps=1,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=False,  # Disable bf16 for CPU
        fp16=False,  # Disable fp16 for CPU
        optim="adamw_torch",
        report_to="none",  # Disable wandb/tensorboard for simple script
        remove_unused_columns=False,
        use_cpu=True,  # Force CPU training
    )
    
    # Create trainer
    # Note: TRL 0.27+ uses 'processing_class' instead of 'tokenizer'
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=formatted_dataset,
        processing_class=tokenizer,
        dataset_text_field="text",
        max_seq_length=2048,  # Reduced for CPU training
        packing=False,  # Don't pack sequences for small dataset
    )
    
    # Train
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    trainer.train()
    
    # Save model
    print("\n" + "=" * 60)
    print("Saving model...")
    print("=" * 60)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print(f"\n✓ Model saved to {OUTPUT_DIR}")
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
