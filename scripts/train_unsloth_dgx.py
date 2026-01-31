#!/usr/bin/env python3
"""
Stage 0 Proof-of-Concept Training with Unsloth on DGX Spark.

Based on NVIDIA DGX Spark Unsloth Playbook:
https://build.nvidia.com/spark/unsloth/instructions
"""

# IMPORTANT: Import unsloth first for optimizations
from unsloth import FastLanguageModel, FastModel

import json
import os
from pathlib import Path

from datasets import Dataset
from trl import SFTTrainer, SFTConfig

# Configuration
MAX_SEQ_LENGTH = 2048
OUTPUT_DIR = "models/stage_0"
DATA_PATH = "data/training/stage_0.jsonl"

# Create output directory
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


def load_training_data() -> Dataset:
    """Load Pramana training data from JSONL."""
    examples = []
    data_path = Path(DATA_PATH)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found at {DATA_PATH}")
    
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            example = json.loads(line)
            # Format for Unsloth: combine instruction and output
            text = f"""### Problem:
{example['instruction']}

### Nyaya Reasoning:
{example['output']}"""
            examples.append({"text": text})
    
    print(f"Loaded {len(examples)} training examples")
    return Dataset.from_list(examples)


def main():
    """Main training function using Unsloth on DGX Spark."""
    print("=" * 60)
    print("Pramana Stage 0 Training with Unsloth on DGX Spark")
    print("=" * 60)
    
    # Check for HuggingFace token
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if hf_token:
        print(f"✓ HF_TOKEN found (length: {len(hf_token)})")
    else:
        print("⚠ No HF_TOKEN found - using non-gated models only")
    
    # Load model using Unsloth's FastModel (optimized for DGX Spark)
    print("\n[1/4] Loading model with Unsloth...")
    
    # Use Gemma 3 4B as recommended by NVIDIA, or Llama 3.2 3B
    model, tokenizer = FastModel.from_pretrained(
        model_name="unsloth/Llama-3.2-3B-Instruct-bnb-4bit",  # 4-bit quantized
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,  # 4-bit quantization for memory efficiency
        load_in_8bit=False,
        full_finetuning=False,  # Use LoRA
        token=hf_token,
    )
    
    # Apply LoRA with Unsloth optimizations
    print("\n[2/4] Applying LoRA with Unsloth optimizations...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=32,  # LoRA rank
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=32,
        lora_dropout=0,  # 0 is optimized by Unsloth
        bias="none",
        use_gradient_checkpointing="unsloth",  # 30% less VRAM
        random_state=42,
        max_seq_length=MAX_SEQ_LENGTH,
        use_rslora=False,
        loftq_config=None,
    )
    
    # Load training data
    print("\n[3/4] Loading training data...")
    dataset = load_training_data()
    
    # Create trainer with Unsloth-optimized settings (TRL 0.24+ API)
    print("\n[4/4] Setting up SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        tokenizer=tokenizer,
        args=SFTConfig(
            max_seq_length=MAX_SEQ_LENGTH,
            per_device_train_batch_size=1,  # Small batch for 5 examples
            gradient_accumulation_steps=4,
            warmup_steps=2,
            max_steps=50,  # ~10 passes through 5 examples
            logging_steps=1,
            output_dir=OUTPUT_DIR,
            optim="adamw_8bit",  # Unsloth-optimized
            learning_rate=2e-5,
            seed=42,
            save_strategy="steps",
            save_steps=25,
            fp16=False,
            bf16=True,
        ),
        dataset_text_field="text",
    )
    
    # Train
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    trainer.train()
    
    # Save the model
    print("\n" + "=" * 60)
    print("Saving model...")
    print("=" * 60)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print(f"\n✓ Model saved to {OUTPUT_DIR}")
    print("\nStage 0 training complete!")
    print("\nTo test the model:")
    print(f"  model, tokenizer = FastLanguageModel.from_pretrained('{OUTPUT_DIR}')")


if __name__ == "__main__":
    main()
