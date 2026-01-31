#!/usr/bin/env python3
"""
Stage 0 Corrected Training Script with Unsloth on DGX Spark.

Implements corrective measures from docs/stage_0_corrective_plan.md:
- LoRA rank: 32 → 64, alpha: 32 → 64
- Explicit format system prompt for Nyaya structure
- Sequence length: 2048 → 4096 tokens
- Epochs: 25 → 10 (with validation monitoring)
- Batch size: 1 → 2
- Validation split: 80/20 train/val
- Format validation callback during training
"""

# IMPORTANT: Import unsloth first for optimizations
from unsloth import FastLanguageModel, FastModel

import json
import os
import re
import math
from pathlib import Path

from datasets import Dataset
from transformers import TrainerCallback
from trl import SFTTrainer, SFTConfig
import torch

# Configuration
MAX_SEQ_LENGTH = 4096  # Increased from 2048
NUM_TRAIN_EPOCHS = 30
USE_CHAT_TEMPLATE = True
OUTPUT_DIR = "models/stage_0_corrected"
DATA_PATH = "data/training/stage_0.jsonl"

# Create output directory
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


# Format enforcement prompt (explicit Nyaya structure requirements)
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

# Required phases for format validation
REQUIRED_PHASES = [
    "samshaya",
    "pramana",
    "pancha avayava",
    "tarka",
    "hetvabhasa",
    "nirnaya",
]


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

    if USE_CHAT_TEMPLATE and hasattr(tokenizer, "apply_chat_template") and getattr(
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


class FormatValidationCallback(TrainerCallback):
    """Callback to validate format adherence during training."""
    
    def __init__(self, tokenizer, sample_problems: list[str], eval_steps: int = 20):
        """Initialize format validation callback.
        
        Args:
            tokenizer: Tokenizer for encoding prompts
            sample_problems: List of sample problem strings for validation
            eval_steps: Evaluate format every N steps
        """
        self.tokenizer = tokenizer
        self.sample_problems = sample_problems
        self.eval_steps = eval_steps
        self.format_adherence_history = []
    
    def _check_phase_presence(self, text: str) -> float:
        """Check how many required phases are present in the output.
        
        Args:
            text: Generated text to check
            
        Returns:
            Fraction of phases found (0.0 to 1.0)
        """
        if not text or not text.strip():
            return 0.0
        
        text_lower = text.lower()
        phases_found = 0
        
        for phase in REQUIRED_PHASES:
            # Check for markdown headers (## Phase Name or # Phase Name)
            pattern = rf"^#+\s+{re.escape(phase)}"
            if re.search(pattern, text_lower, re.MULTILINE | re.IGNORECASE):
                phases_found += 1
        
        return phases_found / len(REQUIRED_PHASES)
    
    def _create_prompt(self, problem: str) -> str:
        """Create inference prompt with format enforcement."""
        user_prompt = build_user_prompt(problem)
        return format_chat_text(
            self.tokenizer, user_prompt, assistant_response=None, add_generation_prompt=True
        )
    
    def on_evaluate(self, args, state, control, model=None, **kwargs):
        """Run format validation after evaluation."""
        if state.global_step % self.eval_steps != 0:
            return
        
        if model is None:
            return
        
        # Generate outputs for sample problems
        model.eval()
        format_scores = []
        
        for problem in self.sample_problems[:2]:  # Test on 2 problems
            try:
                prompt = self._create_prompt(problem)
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=MAX_SEQ_LENGTH,
                ).to(model.device)
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=1024,
                        temperature=0.0,
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
                
                generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
                generated_part = self.tokenizer.decode(
                    generated_tokens, skip_special_tokens=True
                ).strip()
                
                # Check format adherence
                format_score = self._check_phase_presence(generated_part)
                format_scores.append(format_score)
            except Exception as e:
                print(f"Warning: Format validation failed for problem: {e}")
                format_scores.append(0.0)
        
        avg_format_adherence = sum(format_scores) / len(format_scores) if format_scores else 0.0
        self.format_adherence_history.append({
            "step": state.global_step,
            "format_adherence": avg_format_adherence,
        })
        
        print(f"\n[Step {state.global_step}] Format Adherence: {avg_format_adherence:.2%}")
        
        model.train()


def load_training_data(tokenizer) -> Dataset:
    """Load Pramana training data from JSONL with format enforcement."""
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
            
            # Format with explicit format instructions
            user_prompt = build_user_prompt(example["instruction"])
            text = format_chat_text(
                tokenizer,
                user_prompt,
                assistant_response=example["output"],
                add_generation_prompt=False,
            )
            examples.append({"text": text})
    
    print(f"Loaded {len(examples)} training examples")
    return Dataset.from_list(examples)


def main():
    """Main training function using Unsloth on DGX Spark with corrections."""
    print("=" * 60)
    print("Pramana Stage 0 Corrected Training with Unsloth on DGX Spark")
    print("=" * 60)
    
    # Check for HuggingFace token
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if hf_token:
        print(f"✓ HF_TOKEN found (length: {len(hf_token)})")
    else:
        print("⚠ No HF_TOKEN found - using non-gated models only")
    
    # Load model using Unsloth's FastModel (optimized for DGX Spark)
    print("\n[1/5] Loading model with Unsloth...")
    
    # Use Llama 3.2 3B as base model
    model, tokenizer = FastModel.from_pretrained(
        model_name="unsloth/Llama-3.2-3B-Instruct-bnb-4bit",  # 4-bit quantized
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,  # 4-bit quantization for memory efficiency
        load_in_8bit=False,
        full_finetuning=False,  # Use LoRA
        token=hf_token,
    )
    
    # Apply LoRA with increased rank (32 → 64)
    print("\n[2/5] Applying LoRA with increased rank (64)...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=64,  # LoRA rank (doubled from 32)
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=64,  # Match rank (doubled from 32)
        lora_dropout=0,  # 0 is optimized by Unsloth
        bias="none",
        use_gradient_checkpointing="unsloth",  # 30% less VRAM
        random_state=42,
        max_seq_length=MAX_SEQ_LENGTH,
        use_rslora=False,
        loftq_config=None,
    )
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    # Load training data
    print("\n[3/5] Loading training data...")
    dataset = load_training_data(tokenizer)
    
    # Split dataset: 80% train, 20% validation
    print("\n[4/5] Creating train/validation split (80/20)...")
    split_dataset = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    
    print(f"Training examples: {len(train_dataset)}")
    print(f"Validation examples: {len(eval_dataset)}")
    
    # Prepare sample problems for format validation callback
    sample_problems = []
    for example in dataset:
        # Extract problem from text (everything between "### Problem:" and "### Instructions:")
        text = example["text"]
        if "### Problem:" in text and "### Instructions:" in text:
            problem_part = text.split("### Problem:")[1].split("### Instructions:")[0].strip()
            sample_problems.append(problem_part)
    
    # Create format validation callback
    format_callback = FormatValidationCallback(
        tokenizer=tokenizer,
        sample_problems=sample_problems,
        eval_steps=20,  # Evaluate every 20 steps
    )
    
    # Calculate training steps for ~10 epochs
    # With batch_size=2, gradient_accumulation_steps=4, effective batch size = 8
    # For 16 training examples: 16 / 8 = 2 steps per epoch
    # 10 epochs = 20 steps
    num_train_examples = len(train_dataset)
    per_device_batch_size = 2
    gradient_accumulation_steps = 4
    effective_batch_size = per_device_batch_size * gradient_accumulation_steps
    steps_per_epoch = max(1, math.ceil(num_train_examples / effective_batch_size))
    max_steps = steps_per_epoch * NUM_TRAIN_EPOCHS
    
    print(f"\nTraining configuration:")
    print(f"  - Effective batch size: {effective_batch_size}")
    print(f"  - Steps per epoch: {steps_per_epoch}")
    print(f"  - Max steps ({NUM_TRAIN_EPOCHS} epochs): {max_steps}")
    
    # Create trainer with corrected settings
    print("\n[5/5] Setting up SFTTrainer with corrected hyperparameters...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,  # Add validation set
        tokenizer=tokenizer,
        args=SFTConfig(
            max_seq_length=MAX_SEQ_LENGTH,
            per_device_train_batch_size=per_device_batch_size,  # Increased from 1 to 2
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=4,  # Increased from 2
            max_steps=max_steps,
            num_train_epochs=NUM_TRAIN_EPOCHS,  # Avoid None in trainer comparisons
            eval_strategy="steps",  # Evaluate during training
            eval_steps=20,  # Evaluate every 20 steps
            save_strategy="steps",
            save_steps=20,
            logging_steps=5,
            output_dir=OUTPUT_DIR,
            optim="adamw_8bit",  # Unsloth-optimized
            learning_rate=2e-5,
            seed=42,
            fp16=False,
            bf16=True,
            load_best_model_at_end=True,  # Load best checkpoint based on validation loss
            metric_for_best_model="eval_loss",  # Use validation loss
            greater_is_better=False,  # Lower eval_loss is better
        ),
        dataset_text_field="text",
        callbacks=[format_callback],  # Add format validation callback
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
    
    # Print format adherence history
    if format_callback.format_adherence_history:
        print("\n" + "=" * 60)
        print("Format Adherence History:")
        print("=" * 60)
        for entry in format_callback.format_adherence_history:
            print(f"Step {entry['step']}: {entry['format_adherence']:.2%}")
    
    print(f"\n✓ Model saved to {OUTPUT_DIR}")
    print("\nStage 0 corrected training complete!")
    print("\nTo test the model:")
    print(f"  model, tokenizer = FastLanguageModel.from_pretrained('{OUTPUT_DIR}')")


if __name__ == "__main__":
    main()
