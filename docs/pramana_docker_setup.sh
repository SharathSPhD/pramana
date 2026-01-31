#!/bin/bash
# Pramana Project: Nyaya-Tuned Reasoning Engine Docker Setup
# Based on NVIDIA DGX Spark Unsloth Playbook

set -e  # Exit on error

echo "==================================================================="
echo "PRAMANA PROJECT: Docker Environment Setup"
echo "Nyaya-Tuned LLM with GRPO Fine-tuning on DGX Spark"
echo "==================================================================="
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# Step 1: Verify Prerequisites
echo "Step 1: Verifying DGX Spark Prerequisites..."
echo "-----------------------------------------------------------"

if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
    print_status "CUDA Toolkit detected: Version $CUDA_VERSION"
else
    print_error "CUDA Toolkit not found. Please ensure DGX Spark is properly configured."
    exit 1
fi

if command -v nvidia-smi &> /dev/null; then
    print_status "NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    print_error "nvidia-smi not found. GPU not accessible."
    exit 1
fi

# Step 2: Create Project Structure
echo ""
echo "Step 2: Creating Pramana Project Directory Structure..."
echo "-----------------------------------------------------------"

PRAMANA_ROOT="$HOME/pramana-project"
mkdir -p "$PRAMANA_ROOT"/{data,models,scripts,results,configs}
mkdir -p "$PRAMANA_ROOT/data"/{seed_examples,synthetic,validation}
mkdir -p "$PRAMANA_ROOT/models"/{checkpoints,final}
mkdir -p "$PRAMANA_ROOT/scripts"/{training,evaluation,validation}

print_status "Project structure created at: $PRAMANA_ROOT"

# Step 3: Pull Docker Image
echo ""
echo "Step 3: Pulling NVIDIA PyTorch Container..."
echo "-----------------------------------------------------------"
print_warning "This may take several minutes depending on your connection..."

docker pull nvcr.io/nvidia/pytorch:25.09-py3

print_status "Docker image pulled successfully"

# Step 4: Create Dockerfile for Pramana Environment
echo ""
echo "Step 4: Creating Pramana-Specific Dockerfile..."
echo "-----------------------------------------------------------"

cat > "$PRAMANA_ROOT/Dockerfile" << 'DOCKERFILE'
# Pramana Project Dockerfile
# Extends NVIDIA PyTorch with Unsloth, Z3, and custom dependencies

FROM nvcr.io/nvidia/pytorch:25.09-py3

# Set working directory
WORKDIR /workspace/pramana

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    vim \
    tmux \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies for fine-tuning
RUN pip install --no-cache-dir \
    transformers \
    peft \
    "datasets==4.3.0" \
    "trl==0.19.1" \
    hf_transfer

# Install Unsloth
RUN pip install --no-deps unsloth unsloth_zoo

# Install bitsandbytes
RUN pip install --no-deps bitsandbytes

# Install Z3 SMT Solver for neuro-symbolic validation
RUN pip install --no-cache-dir z3-solver

# Install additional dependencies for Pramana project
RUN pip install --no-cache-dir \
    wandb \
    tensorboard \
    jupyter \
    pandas \
    numpy \
    scikit-learn \
    pyyaml \
    tqdm \
    vllm \
    accelerate

# Install Sanskrit NLP tools (optional, for advanced Nyaya analysis)
RUN pip install --no-cache-dir indic-transliteration

# Set environment variables
ENV HF_HOME=/workspace/pramana/hf_cache
ENV WANDB_PROJECT=pramana-nyaya-tuning
ENV TOKENIZERS_PARALLELISM=false

# Create directories
RUN mkdir -p /workspace/pramana/{data,models,scripts,results,configs,hf_cache}

# Set up Jupyter
RUN pip install --no-cache-dir ipykernel ipywidgets

WORKDIR /workspace/pramana

CMD ["/bin/bash"]
DOCKERFILE

print_status "Dockerfile created at: $PRAMANA_ROOT/Dockerfile"

# Step 5: Build Custom Docker Image
echo ""
echo "Step 5: Building Pramana Docker Image..."
echo "-----------------------------------------------------------"
print_warning "This will take 5-10 minutes..."

cd "$PRAMANA_ROOT"
docker build -t pramana-unsloth:latest .

print_status "Pramana Docker image built successfully"

# Step 6: Create Docker Run Script
echo ""
echo "Step 6: Creating Docker Run Scripts..."
echo "-----------------------------------------------------------"

cat > "$PRAMANA_ROOT/run_pramana_container.sh" << 'RUNSCRIPT'
#!/bin/bash
# Launch Pramana Docker container with GPU support

PRAMANA_ROOT="$HOME/pramana-project"

docker run --gpus all \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --shm-size=32g \
    -it \
    --rm \
    -v "$PRAMANA_ROOT/data":/workspace/pramana/data \
    -v "$PRAMANA_ROOT/models":/workspace/pramana/models \
    -v "$PRAMANA_ROOT/scripts":/workspace/pramana/scripts \
    -v "$PRAMANA_ROOT/results":/workspace/pramana/results \
    -v "$PRAMANA_ROOT/configs":/workspace/pramana/configs \
    -v "$HOME/.cache/huggingface":/workspace/pramana/hf_cache \
    -p 8888:8888 \
    -p 6006:6006 \
    --name pramana-dev \
    pramana-unsloth:latest
RUNSCRIPT

chmod +x "$PRAMANA_ROOT/run_pramana_container.sh"

cat > "$PRAMANA_ROOT/run_jupyter.sh" << 'JUPYSCRIPT'
#!/bin/bash
# Launch Pramana container with Jupyter

PRAMANA_ROOT="$HOME/pramana-project"

docker run --gpus all \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --shm-size=32g \
    -it \
    --rm \
    -v "$PRAMANA_ROOT/data":/workspace/pramana/data \
    -v "$PRAMANA_ROOT/models":/workspace/pramana/models \
    -v "$PRAMANA_ROOT/scripts":/workspace/pramana/scripts \
    -v "$PRAMANA_ROOT/results":/workspace/pramana/results \
    -v "$PRAMANA_ROOT/configs":/workspace/pramana/configs \
    -v "$HOME/.cache/huggingface":/workspace/pramana/hf_cache \
    -p 8888:8888 \
    -p 6006:6006 \
    --name pramana-jupyter \
    pramana-unsloth:latest \
    jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
JUPYSCRIPT

chmod +x "$PRAMANA_ROOT/run_jupyter.sh"

print_status "Run scripts created"

# Step 7: Create Initial Configuration Files
echo ""
echo "Step 7: Creating Initial Project Configuration..."
echo "-----------------------------------------------------------"

cat > "$PRAMANA_ROOT/configs/stage_zero_config.yaml" << 'YAMLCONFIG'
# Pramana Stage Zero: Proof of Concept Configuration
# 5-10 manual examples, test basic Nyaya structure learning

experiment_name: "pramana_stage_zero"
stage: 0

# Model Configuration
model:
  base_model: "unsloth/Meta-Llama-3.1-8B-bnb-4bit"
  load_in_4bit: true
  lora_config:
    r: 64
    lora_alpha: 16
    target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    lora_dropout: 0.05
    bias: "none"
    task_type: "CAUSAL_LM"

# Dataset Configuration
data:
  seed_examples_path: "/workspace/pramana/data/seed_examples/stage_zero"
  num_examples: 5
  problem_types: ["constraint_satisfaction"]
  train_test_split: 0.8
  
# Training Configuration
training:
  output_dir: "/workspace/pramana/models/checkpoints/stage_zero"
  num_train_epochs: 10
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 4
  learning_rate: 2.0e-4
  warmup_steps: 10
  logging_steps: 1
  save_steps: 50
  max_seq_length: 4096
  
# Nyaya Structure Validation
nyaya_validation:
  required_phases: ["samshaya", "pramana", "pancha_avayava", "tarka", "hetvabhasa", "nirnaya"]
  strict_format: true
  
# Evaluation
evaluation:
  metrics: ["format_adherence", "reasoning_completeness", "answer_accuracy"]
  validation_examples: 1
YAMLCONFIG

print_status "Stage Zero configuration created"

cat > "$PRAMANA_ROOT/configs/grpo_stage_three_config.yaml" << 'GRPOCONFIG'
# Pramana Stage Three: GRPO Reinforcement Learning Configuration

experiment_name: "pramana_grpo_stage_three"
stage: 3

# Model Configuration
model:
  base_model: "unsloth/Meta-Llama-3.1-8B-bnb-4bit"
  load_in_4bit: true
  use_cache: false

# GRPO Configuration
grpo:
  beta: 0.01  # KL penalty coefficient
  num_generations: 4  # Responses per prompt (must be > 2)
  epsilon: 0.2  # Clipping value for PPO-style objective
  gamma: 1.0  # Discount factor
  lam: 0.95  # GAE lambda
  
  # Custom Nyaya Reward Functions
  reward_functions:
    - name: "nyaya_structure_completeness"
      weight: 0.3
      description: "Rewards presence of all six Nyaya phases"
      
    - name: "logical_consistency"
      weight: 0.25
      description: "Z3 verification of logical soundness"
      
    - name: "hetvabhasa_detection"
      weight: 0.2
      description: "Correct identification of fallacies"
      
    - name: "pramana_appropriateness"
      weight: 0.15
      description: "Correct application of knowledge sources"
      
    - name: "answer_correctness"
      weight: 0.1
      description: "Final answer matches ground truth"

# Training Configuration
training:
  output_dir: "/workspace/pramana/models/checkpoints/stage_three_grpo"
  num_iterations: 300  # GRPO iterations
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8
  learning_rate: 5.0e-6
  warmup_steps: 50
  logging_steps: 5
  save_steps: 50
  max_seq_length: 6144

# Validation
validation:
  eval_every: 25
  num_eval_examples: 20
GRPOCONFIG

print_status "GRPO Stage Three configuration created"

# Step 8: Create Test Script
echo ""
echo "Step 8: Creating Unsloth Validation Script..."
echo "-----------------------------------------------------------"

cat > "$PRAMANA_ROOT/scripts/test_unsloth.py" << 'PYTHONTEST'
"""
Pramana Project: Unsloth Installation Validation
Tests basic fine-tuning capability on DGX Spark
"""

from unsloth import FastLanguageModel
import torch

def test_unsloth_installation():
    print("=" * 70)
    print("PRAMANA PROJECT: Testing Unsloth Installation")
    print("=" * 70)
    
    # Test 1: Load a small model
    print("\n[Test 1] Loading Llama 3.1 8B model with 4-bit quantization...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Meta-Llama-3.1-8B-bnb-4bit",
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    print("✓ Model loaded successfully!")
    
    # Test 2: Add LoRA adapters
    print("\n[Test 2] Adding LoRA adapters for efficient fine-tuning...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    print("✓ LoRA adapters added successfully!")
    
    # Test 3: Quick inference test
    print("\n[Test 3] Testing inference capability...")
    alpaca_prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{}

### Response:
{}"""
    
    FastLanguageModel.for_inference(model)
    inputs = tokenizer([alpaca_prompt.format("What is Nyaya Darshan?", "")], return_tensors="pt").to("cuda")
    
    outputs = model.generate(**inputs, max_new_tokens=64, use_cache=True)
    decoded = tokenizer.batch_decode(outputs)
    print("✓ Inference test passed!")
    print(f"\nSample output: {decoded[0][:200]}...")
    
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED! Unsloth is ready for Pramana project.")
    print("=" * 70)

if __name__ == "__main__":
    test_unsloth_installation()
PYTHONTEST

print_status "Test script created"

# Step 9: Create README
echo ""
echo "Step 9: Creating Project README..."
echo "-----------------------------------------------------------"

cat > "$PRAMANA_ROOT/README.md" << 'READMEMD'
# Pramana Project: Nyaya-Tuned Reasoning Engine

Building epistemic infrastructure for trustworthy AI reasoning using 2,500-year-old Nyaya Darshan principles.

## Project Structure

```
pramana-project/
├── data/
│   ├── seed_examples/     # Manually created Nyaya examples
│   ├── synthetic/         # LLM-generated examples
│   └── validation/        # Test cases
├── models/
│   ├── checkpoints/       # Training checkpoints
│   └── final/            # Production models
├── scripts/
│   ├── training/         # Fine-tuning scripts
│   ├── evaluation/       # Evaluation harness
│   └── validation/       # Z3 validation tools
├── results/              # Experiment results
└── configs/              # Training configurations
```

## Quick Start

### 1. Launch Pramana Container

```bash
cd ~/pramana-project
./run_pramana_container.sh
```

### 2. Validate Installation

```bash
python scripts/test_unsloth.py
```

### 3. Start Stage Zero

Create 5 manual Nyaya examples in `data/seed_examples/stage_zero/`, then:

```bash
python scripts/training/stage_zero_finetune.py --config configs/stage_zero_config.yaml
```

## Development Workflow

### Interactive Development
```bash
./run_jupyter.sh
# Access Jupyter at http://localhost:8888
```

### Training with Monitoring
```bash
# Terminal 1: Start TensorBoard
tensorboard --logdir results/ --port 6006

# Terminal 2: Run training
python scripts/training/train.py --config configs/stage_zero_config.yaml
```

## Stage Implementation

- **Stage Zero** (2 weeks): 5-10 examples, format validation
- **Stage One** (8-10 weeks): 50 examples, minimum viable reasoner
- **Stage Two** (8 weeks): Synthetic scaling to 200-500 examples
- **Stage Three** (8-12 weeks): GRPO reinforcement learning
- **Stage Four** (variable): Deployment & optimization

## Key Features

- ✅ Unsloth for efficient fine-tuning on DGX Spark
- ✅ Z3 SMT solver for neuro-symbolic validation
- ✅ GRPO for custom Nyaya reward functions
- ✅ Full experiment tracking with W&B and TensorBoard
- ✅ Containerized reproducible environment

## References

- Nyaya Sutras (Gautama Maharishi, ~500 BCE)
- Tattvacintāmaṇi (Gaṅgeśa Upādhyāya, 1325 CE)
- DeepSeek-R1: Incentivizing Reasoning via RL (2025)
- Unsloth Documentation: https://docs.unsloth.ai/

---

**विवेकज्ञानं मोक्षः** (Viveka-jñānaṃ mokṣaḥ)  
*"Discriminative wisdom leads to liberation"*
READMEMD

print_status "README created"

# Final Summary
echo ""
echo "==================================================================="
echo "SETUP COMPLETE!"
echo "==================================================================="
echo ""
echo "Project location: $PRAMANA_ROOT"
echo ""
echo "Next steps:"
echo "  1. Launch container: cd $PRAMANA_ROOT && ./run_pramana_container.sh"
echo "  2. Test installation: python scripts/test_unsloth.py"
echo "  3. Create seed examples in: data/seed_examples/stage_zero/"
echo "  4. Begin Stage Zero training"
echo ""
echo "Helpful commands:"
echo "  - Start Jupyter:    ./run_jupyter.sh"
echo "  - View logs:        docker logs pramana-dev"
echo "  - Stop container:   docker stop pramana-dev"
echo ""
print_status "Ready to begin Pramana project implementation!"
echo ""
