# Pramana Project: Architecture Recommendations

**Date**: 2025-01-31  
**Status**: Architectural Analysis & Recommendations  
**Context**: Staged ML research project (4.5 months) with Docker-based development on DGX Spark

---

## Executive Summary

This document provides concrete architectural recommendations for the Pramana project's five critical design areas:

1. **Project Structure Pattern**: Stage-aware directory organization supporting research evolution
2. **Configuration Management Pattern**: Hierarchical configs with stage-specific overrides
3. **Data Pipeline Pattern**: Markdown → training format with validation gates
4. **Evaluation Framework Pattern**: Three-tier evaluation (automated, LLM judge, manual)
5. **Training Abstraction Level**: Moderate abstraction balancing flexibility and speed

Each recommendation includes specific file paths, class names, and implementation patterns aligned with the existing codebase structure.

---

## 1. Project Structure Pattern

### Recommendation: **Stage-Aware Research Structure with Immutable Data Versions**

**Rationale**: ML research projects need to evolve through stages while maintaining reproducibility. The structure must support:
- Stage progression (POC → MVP → Scaling → RL)
- Data versioning (immutable training sets per stage)
- Experiment tracking (multiple attempts per stage)
- Reproducibility (exact configs, seeds, data versions)

### Proposed Structure

```
pramana/
├── pramana/                          # Main Python package
│   ├── __init__.py
│   ├── config/                       # Configuration management
│   │   ├── __init__.py
│   │   ├── base.py                   # BaseConfig class
│   │   ├── stage_config.py           # StageConfigLoader
│   │   └── validators.py             # Config validation
│   ├── data/                         # Data pipeline modules
│   │   ├── __init__.py
│   │   ├── parsers.py                # MarkdownParser, NyayaParser
│   │   ├── validators.py            # NyayaStructureValidator
│   │   ├── converters.py            # MarkdownToTrainingConverter
│   │   └── versioning.py            # DataVersionManager
│   ├── training/                     # Training abstractions
│   │   ├── __init__.py
│   │   ├── base_trainer.py          # BaseTrainer abstract class
│   │   ├── sft_trainer.py           # SupervisedFineTuningTrainer
│   │   ├── grpo_trainer.py          # GRPOTrainer
│   │   └── callbacks.py             # Training callbacks
│   ├── evaluation/                   # Evaluation framework
│   │   ├── __init__.py
│   │   ├── tier1_validator.py       # Tier1StructuralValidator
│   │   ├── tier2_judge.py           # Tier2LLMJudge
│   │   ├── tier3_reviewer.py        # Tier3ManualReviewer
│   │   ├── metrics.py               # NyayaMetrics, FormatMetrics
│   │   └── benchmark_runner.py      # BenchmarkRunner
│   ├── verification/                 # Z3 integration
│   │   ├── __init__.py
│   │   ├── z3_solver.py             # Z3Solver wrapper
│   │   ├── autoformalizer.py        # NyayaToSMTLibConverter
│   │   └── consistency_checker.py   # LogicalConsistencyChecker
│   └── utils/                       # Shared utilities
│       ├── __init__.py
│       ├── logging.py               # Structured logging
│       └── experiment_tracking.py   # W&B/TensorBoard helpers
│
├── data/                             # Data directory (git-tracked)
│   ├── seed_examples/
│   │   ├── stage_zero/
│   │   │   └── v1.0/                # Immutable version
│   │   │       ├── problem_01.md
│   │   │       ├── problem_02.md
│   │   │       └── .dataversion     # Version metadata
│   │   ├── stage_one/
│   │   │   └── v1.0/
│   │   │       ├── constraint_satisfaction/
│   │   │       ├── boolean_sat/
│   │   │       └── multi_step_deduction/
│   │   └── stage_two_synthetic/
│   │       └── v1.0/
│   │           ├── batch_001/
│   │           └── tier1_passed/
│   ├── validation/
│   │   ├── held_out/                # Never-seen test cases
│   │   └── benchmarks/              # LogicBench, ProntoQA
│   └── evaluation/
│       └── adversarial/             # Hetvabhasa test cases
│
├── models/                           # Model artifacts (git-ignored)
│   ├── checkpoints/
│   │   ├── stage_zero/
│   │   │   ├── attempt_01/          # Experiment tracking
│   │   │   │   ├── checkpoint-100/
│   │   │   │   └── checkpoint-200/
│   │   │   └── attempt_02/
│   │   ├── stage_one/
│   │   └── stage_three_grpo/
│   └── final/                       # Production-ready models
│
├── configs/                         # YAML configuration files
│   ├── base.yaml                    # Shared defaults
│   ├── stage_zero.yaml
│   ├── stage_one.yaml
│   ├── stage_two.yaml
│   └── stage_three_grpo.yaml
│
├── scripts/                         # Executable scripts
│   ├── data/
│   │   ├── parse_markdown.py       # CLI: Parse markdown → JSON
│   │   ├── validate_structure.py   # CLI: Validate Nyaya structure
│   │   └── generate_synthetic.py   # CLI: Synthetic generation
│   ├── training/
│   │   ├── train_stage_zero.py     # CLI: Stage 0 training
│   │   ├── train_stage_one.py      # CLI: Stage 1 training
│   │   └── train_grpo.py           # CLI: GRPO training
│   └── evaluation/
│       ├── run_tier1.py            # CLI: Tier 1 validation
│       ├── run_tier2.py            # CLI: Tier 2 LLM judge
│       └── run_benchmarks.py       # CLI: Benchmark evaluation
│
├── results/                         # Experiment results (git-ignored)
│   ├── experiments/                # W&B/TensorBoard logs
│   ├── evaluations/                # Evaluation reports (JSON)
│   └── analysis/                   # Error analysis, ablations
│
├── tests/                           # Unit tests
│   ├── test_data_parsers.py
│   ├── test_validators.py
│   └── test_evaluation.py
│
├── docs/                            # Documentation
│   ├── architecture_recommendations.md  # This file
│   ├── plans/
│   └── ...
│
├── Dockerfile
├── requirements.txt                 # Pinned dependencies
├── pyproject.toml                  # Python package config
└── README.md
```

### Key Design Decisions

**1. Package Structure (`pramana/`)**
- **Rationale**: Enables `from pramana.data import NyayaParser` imports
- **Benefit**: Reusable modules, testable components, clear API boundaries
- **File**: `pramana/__init__.py` exports main classes

**2. Stage-Based Data Organization**
- **Pattern**: `data/seed_examples/stage_{N}/v{M}.{m}/`
- **Rationale**: Immutable versions enable reproducibility. Each training run references exact data version.
- **Implementation**: `pramana/data/versioning.py::DataVersionManager`

**3. Experiment Tracking in Checkpoints**
- **Pattern**: `models/checkpoints/stage_{N}/attempt_{M}/checkpoint-{step}/`
- **Rationale**: Multiple attempts per stage (hyperparameter tuning) without losing history
- **Metadata**: Each checkpoint includes `experiment_config.yaml` and `git_commit.txt`

**4. Config Separation**
- **Pattern**: `configs/base.yaml` + `configs/stage_{N}.yaml` (inheritance)
- **Rationale**: Shared defaults (base) + stage-specific overrides
- **Implementation**: `pramana/config/stage_config.py::StageConfigLoader`

---

## 2. Configuration Management Pattern

### Recommendation: **Hierarchical YAML Configs with Python Validation**

**Rationale**: ML experiments need:
- Reproducibility (exact configs saved with checkpoints)
- Flexibility (override via CLI/env vars)
- Validation (catch errors before training starts)
- Stage-specific defaults (avoid repetition)

### Implementation Pattern

**Base Configuration** (`configs/base.yaml`):
```yaml
# configs/base.yaml
# Shared defaults across all stages

project:
  name: "pramana"
  version: "0.1.0"

model:
  base_model: "unsloth/Meta-Llama-3.1-8B-bnb-4bit"
  load_in_4bit: true
  max_seq_length: 4096

lora:
  r: 64
  lora_alpha: 16
  target_modules:
    - "q_proj"
    - "k_proj"
    - "v_proj"
    - "o_proj"
    - "gate_proj"
    - "up_proj"
    - "down_proj"
  lora_dropout: 0.05
  bias: "none"

training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16
  learning_rate: 2.0e-5
  warmup_ratio: 0.1
  lr_scheduler_type: "cosine"
  weight_decay: 0.01
  max_grad_norm: 0.3
  logging_steps: 10
  save_steps: 100

experiment_tracking:
  wandb_project: "pramana-nyaya-tuning"
  tensorboard_logdir: "results/experiments"

paths:
  data_root: "data"
  models_root: "models"
  results_root: "results"
```

**Stage-Specific Override** (`configs/stage_zero.yaml`):
```yaml
# configs/stage_zero.yaml
# Extends base.yaml, overrides for Stage 0

extends: "base.yaml"

experiment:
  name: "pramana_stage_zero"
  stage: 0
  description: "Proof of concept with 5 examples"

data:
  seed_examples_path: "data/seed_examples/stage_zero/v1.0"
  num_examples: 5
  train_test_split: 0.8
  problem_types:
    - "constraint_satisfaction"

training:
  num_train_epochs: 10
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 4
  learning_rate: 2.0e-4  # Higher LR for small dataset
  warmup_steps: 10
  logging_steps: 1
  save_steps: 50

evaluation:
  metrics:
    - "format_adherence"
    - "reasoning_completeness"
    - "answer_accuracy"
  validation_examples: 1

nyaya_validation:
  required_phases:
    - "samshaya"
    - "pramana"
    - "pancha_avayava"
    - "tarka"
    - "hetvabhasa"
    - "nirnaya"
  strict_format: true
```

**Python Config Classes** (`pramana/config/base.py`):
```python
# pramana/config/base.py

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path
import yaml
from pydantic import BaseModel, Field, validator

class ModelConfig(BaseModel):
    """Model configuration"""
    base_model: str
    load_in_4bit: bool = True
    max_seq_length: int = 4096

class LoRAConfig(BaseModel):
    """LoRA adapter configuration"""
    r: int = 64
    lora_alpha: int = 16
    target_modules: List[str] = Field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    lora_dropout: float = 0.05
    bias: str = "none"
    
    @validator('r')
    def validate_r(cls, v):
        assert 8 <= v <= 256, "LoRA rank must be between 8 and 256"
        return v

class TrainingConfig(BaseModel):
    """Training hyperparameters"""
    num_train_epochs: int
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2.0e-5
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 0.01
    max_grad_norm: float = 0.3
    logging_steps: int = 10
    save_steps: int = 100
    output_dir: str = "models/checkpoints"
    
    @validator('learning_rate')
    def validate_lr(cls, v):
        assert 1e-6 <= v <= 1e-3, "Learning rate out of reasonable range"
        return v

class DataConfig(BaseModel):
    """Data configuration"""
    seed_examples_path: str
    num_examples: Optional[int] = None
    train_test_split: float = 0.8
    problem_types: List[str] = Field(default_factory=list)

class ExperimentConfig(BaseModel):
    """Complete experiment configuration"""
    experiment: Dict[str, Any]
    model: ModelConfig
    lora: LoRAConfig
    training: TrainingConfig
    data: DataConfig
    evaluation: Dict[str, Any] = Field(default_factory=dict)
    nyaya_validation: Dict[str, Any] = Field(default_factory=dict)
    experiment_tracking: Dict[str, str] = Field(default_factory=dict)
    paths: Dict[str, str] = Field(default_factory=dict)
    
    @classmethod
    def from_yaml(cls, config_path: Path, overrides: Optional[Dict] = None):
        """Load config from YAML with inheritance support"""
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
        
        # Handle 'extends' for config inheritance
        if 'extends' in config_dict:
            base_path = config_path.parent / config_dict['extends']
            base_config = cls.from_yaml(base_path)
            # Merge: base values, then stage-specific overrides
            merged = {**base_config.dict(), **config_dict}
            merged.pop('extends', None)
        else:
            merged = config_dict
        
        # Apply CLI/env overrides
        if overrides:
            merged = cls._deep_merge(merged, overrides)
        
        return cls(**merged)
    
    @staticmethod
    def _deep_merge(base: dict, overrides: dict) -> dict:
        """Recursively merge dictionaries"""
        result = base.copy()
        for key, value in overrides.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = cls._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    def save(self, output_path: Path):
        """Save config to YAML for reproducibility"""
        with open(output_path, 'w') as f:
            yaml.dump(self.dict(), f, default_flow_style=False)
```

**Config Loader** (`pramana/config/stage_config.py`):
```python
# pramana/config/stage_config.py

from pathlib import Path
from typing import Optional
from .base import ExperimentConfig

class StageConfigLoader:
    """Loads and validates stage-specific configurations"""
    
    STAGE_CONFIGS = {
        0: "configs/stage_zero.yaml",
        1: "configs/stage_one.yaml",
        2: "configs/stage_two.yaml",
        3: "configs/stage_three_grpo.yaml",
    }
    
    @classmethod
    def load_stage_config(
        cls,
        stage: int,
        config_dir: Path = Path("configs"),
        overrides: Optional[dict] = None
    ) -> ExperimentConfig:
        """Load configuration for a specific stage"""
        if stage not in cls.STAGE_CONFIGS:
            raise ValueError(f"Unknown stage: {stage}")
        
        config_path = config_dir / cls.STAGE_CONFIGS[stage]
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        
        config = ExperimentConfig.from_yaml(config_path, overrides=overrides)
        
        # Validate stage matches
        if config.experiment.get('stage') != stage:
            raise ValueError(f"Config stage mismatch: expected {stage}, got {config.experiment.get('stage')}")
        
        return config
```

**Usage Pattern**:
```python
# scripts/training/train_stage_zero.py

from pramana.config import StageConfigLoader
from pramana.training import SupervisedFineTuningTrainer

# Load config with CLI overrides
config = StageConfigLoader.load_stage_config(
    stage=0,
    overrides={"training": {"num_train_epochs": 15}}  # Override epochs
)

# Save config with checkpoint for reproducibility
trainer = SupervisedFineTuningTrainer(config)
trainer.train()
```

### Key Design Decisions

**1. YAML + Pydantic Hybrid**
- **YAML**: Human-readable, Git-friendly, easy to edit
- **Pydantic**: Runtime validation, type safety, IDE autocomplete
- **Benefit**: Best of both worlds

**2. Config Inheritance**
- **Pattern**: `extends: "base.yaml"` in stage configs
- **Rationale**: DRY principle, shared defaults, stage-specific overrides
- **Implementation**: `ExperimentConfig.from_yaml()` handles merging

**3. Override Support**
- **Pattern**: CLI args → env vars → config file → base defaults
- **Rationale**: Flexibility for experimentation without editing files
- **Usage**: `python train.py --training.num_train_epochs 15`

**4. Config Persistence**
- **Pattern**: Save `experiment_config.yaml` with each checkpoint
- **Rationale**: Reproducibility - exact config used for training
- **Implementation**: `trainer.save_config(checkpoint_dir)`

---

## 3. Data Pipeline Pattern

### Recommendation: **Pipeline with Validation Gates and Versioning**

**Rationale**: Markdown → training format conversion needs:
- Structured parsing (YAML frontmatter + markdown sections)
- Validation gates (catch errors before training)
- Versioning (immutable datasets per stage)
- Format conversion (markdown → HuggingFace dataset format)

### Implementation Pattern

**Markdown Parser** (`pramana/data/parsers.py`):
```python
# pramana/data/parsers.py

import frontmatter
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
import re

@dataclass
class NyayaExample:
    """Structured representation of a Nyaya training example"""
    id: str
    problem: str
    problem_type: str
    ground_truth: str
    metadata: Dict[str, Any]
    samshaya: str
    pramana: Dict[str, Any]
    pancha_avayava: List[Dict[str, Any]]
    tarka: str
    hetvabhasa: Dict[str, Any]
    nirnaya: str

class MarkdownParser:
    """Parses markdown files with YAML frontmatter into NyayaExample"""
    
    REQUIRED_SECTIONS = [
        "Problem",
        "Samshaya",
        "Pramana",
        "Pancha Avayava",
        "Tarka",
        "Hetvabhasa",
        "Nirnaya"
    ]
    
    def parse(self, file_path: Path) -> NyayaExample:
        """Parse markdown file to NyayaExample"""
        with open(file_path) as f:
            post = frontmatter.load(f)
        
        metadata = post.metadata
        content = post.content
        
        # Extract problem statement
        problem = self._extract_section(content, "Problem")
        
        # Extract Nyaya phases
        samshaya = self._extract_section(content, "Samshaya")
        pramana = self._extract_pramana(content)
        pancha_avayava = self._extract_pancha_avayava(content)
        tarka = self._extract_section(content, "Tarka")
        hetvabhasa = self._extract_hetvabhasa(content)
        nirnaya = self._extract_section(content, "Nirnaya")
        
        return NyayaExample(
            id=metadata.get("id", file_path.stem),
            problem=problem,
            problem_type=metadata.get("problem_type", "unknown"),
            ground_truth=metadata.get("ground_truth", ""),
            metadata=metadata,
            samshaya=samshaya,
            pramana=pramana,
            pancha_avayava=pancha_avayava,
            tarka=tarka,
            hetvabhasa=hetvabhasa,
            nirnaya=nirnaya
        )
    
    def _extract_section(self, content: str, section_name: str) -> str:
        """Extract markdown section by header"""
        pattern = rf"##\s+{re.escape(section_name)}.*?\n(.*?)(?=##|\Z)"
        match = re.search(pattern, content, re.DOTALL)
        if not match:
            raise ValueError(f"Missing required section: {section_name}")
        return match.group(1).strip()
    
    def _extract_pramana(self, content: str) -> Dict[str, Any]:
        """Extract Pramana section with YAML subsections"""
        pramana_section = self._extract_section(content, "Pramana")
        
        # Extract Pratyaksha, Anumana, Upamana, Shabda
        pramana = {}
        for source_type in ["Pratyaksha", "Anumana", "Upamana", "Shabda"]:
            pattern = rf"###\s+{re.escape(source_type)}.*?\n(.*?)(?=###|\Z)"
            match = re.search(pattern, pramana_section, re.DOTALL)
            if match:
                # Try to parse as YAML if it's a code block
                yaml_match = re.search(r"```yaml\n(.*?)\n```", match.group(1), re.DOTALL)
                if yaml_match:
                    import yaml
                    pramana[source_type.lower()] = yaml.safe_load(yaml_match.group(1))
                else:
                    pramana[source_type.lower()] = match.group(1).strip()
        
        return pramana
    
    def _extract_pancha_avayava(self, content: str) -> List[Dict[str, Any]]:
        """Extract all Pancha Avayava syllogisms"""
        avayava_section = self._extract_section(content, "Pancha Avayava")
        
        # Find all syllogism blocks (### Syllogism N:)
        syllogism_pattern = r"###\s+Syllogism\s+\d+.*?\n(.*?)(?=###|\Z)"
        matches = re.finditer(syllogism_pattern, avayava_section, re.DOTALL)
        
        syllogisms = []
        for match in matches:
            syllogism_text = match.group(1)
            syllogism = {
                "pratijna": self._extract_component(syllogism_text, "Pratijna"),
                "hetu": self._extract_component(syllogism_text, "Hetu"),
                "udaharana": self._extract_component(syllogism_text, "Udaharana"),
                "upanaya": self._extract_component(syllogism_text, "Upanaya"),
                "nigamana": self._extract_component(syllogism_text, "Nigamana"),
            }
            syllogisms.append(syllogism)
        
        return syllogisms
    
    def _extract_component(self, text: str, component: str) -> str:
        """Extract component from syllogism text"""
        pattern = rf"\*\*{re.escape(component)}.*?\*\*:\s*(.*?)(?=\*\*|\Z)"
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else ""
    
    def _extract_hetvabhasa(self, content: str) -> Dict[str, Any]:
        """Extract Hetvabhasa section (YAML format)"""
        hetvabhasa_section = self._extract_section(content, "Hetvabhasa")
        
        # Extract YAML code block
        yaml_match = re.search(r"```yaml\n(.*?)\n```", hetvabhasa_section, re.DOTALL)
        if yaml_match:
            import yaml
            return yaml.safe_load(yaml_match.group(1))
        return {}
```

**Structure Validator** (`pramana/data/validators.py`):
```python
# pramana/data/validators.py

from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from .parsers import NyayaExample

@dataclass
class ValidationResult:
    """Result of Nyaya structure validation"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    phase_completeness: Dict[str, bool]

class NyayaStructureValidator:
    """Validates Nyaya example structure and content"""
    
    REQUIRED_PRAMANA_TYPES = ["pratyaksha", "anumana", "upamana", "shabda"]
    REQUIRED_AVAYAVA_COMPONENTS = ["pratijna", "hetu", "udaharana", "upanaya", "nigamana"]
    REQUIRED_HETVABHASA_CHECKS = [
        "savyabhichara", "viruddha", "prakaranasama", "sadhyasama", "kalaatita"
    ]
    
    def validate(self, example: NyayaExample) -> ValidationResult:
        """Validate Nyaya example structure"""
        errors = []
        warnings = []
        phase_completeness = {}
        
        # Check Pramana completeness
        pramana_valid, pramana_errors = self._validate_pramana(example.pramana)
        phase_completeness["pramana"] = pramana_valid
        errors.extend(pramana_errors)
        
        # Check Pancha Avayava completeness
        avayava_valid, avayava_errors = self._validate_pancha_avayava(example.pancha_avayava)
        phase_completeness["pancha_avayava"] = avayava_valid
        errors.extend(avayava_errors)
        
        # Check Hetvabhasa completeness
        hetvabhasa_valid, hetvabhasa_errors = self._validate_hetvabhasa(example.hetvabhasa)
        phase_completeness["hetvabhasa"] = hetvabhasa_valid
        errors.extend(hetvabhasa_errors)
        
        # Check Udaharana has universal rule
        udaharana_warnings = self._check_udaharana_universal_rules(example.pancha_avayava)
        warnings.extend(udaharana_warnings)
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            phase_completeness=phase_completeness
        )
    
    def _validate_pramana(self, pramana: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate Pramana section has all 4 types"""
        errors = []
        for pramana_type in self.REQUIRED_PRAMANA_TYPES:
            if pramana_type not in pramana or not pramana[pramana_type]:
                errors.append(f"Missing or empty {pramana_type} in Pramana")
        return len(errors) == 0, errors
    
    def _validate_pancha_avayava(self, syllogisms: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
        """Validate Pancha Avayava has complete syllogisms"""
        errors = []
        if len(syllogisms) == 0:
            errors.append("No Pancha Avayava syllogisms found")
        
        for i, syllogism in enumerate(syllogisms):
            for component in self.REQUIRED_AVAYAVA_COMPONENTS:
                if component not in syllogism or not syllogism[component]:
                    errors.append(f"Syllogism {i+1} missing {component}")
        
        return len(errors) == 0, errors
    
    def _validate_hetvabhasa(self, hetvabhasa: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate Hetvabhasa checks all 5 fallacy types"""
        errors = []
        fallacy_checks = hetvabhasa.get("fallacy_checks", {})
        for fallacy_type in self.REQUIRED_HETVABHASA_CHECKS:
            if fallacy_type not in fallacy_checks:
                errors.append(f"Missing {fallacy_type} check in Hetvabhasa")
        return len(errors) == 0, errors
    
    def _check_udaharana_universal_rules(self, syllogisms: List[Dict[str, Any]]) -> List[str]:
        """Check if Udaharana contains universal rules"""
        warnings = []
        for i, syllogism in enumerate(syllogisms):
            udaharana = syllogism.get("udaharana", "")
            if "wherever" not in udaharana.lower() and "universal" not in udaharana.lower():
                warnings.append(
                    f"Syllogism {i+1} Udaharana may lack universal rule "
                    "(should contain 'Wherever X, there is Y' structure)"
                )
        return warnings
```

**Training Format Converter** (`pramana/data/converters.py`):
```python
# pramana/data/converters.py

from typing import List, Dict, Any
from datasets import Dataset
from .parsers import NyayaExample

class MarkdownToTrainingConverter:
    """Converts NyayaExample to HuggingFace training format"""
    
    SYSTEM_PROMPT = """You are a reasoning engine that solves logical problems using Nyaya Darshan methodology. Apply the systematic six-phase approach:
1. Samshaya (Doubt Analysis)
2. Pramana (Evidence Sources)
3. Pancha Avayava (Five-Member Syllogism)
4. Tarka (Counterfactual Testing)
5. Hetvabhasa (Fallacy Detection)
6. Nirnaya (Definitive Conclusion)"""
    
    def convert(self, examples: List[NyayaExample]) -> Dataset:
        """Convert list of NyayaExamples to HuggingFace Dataset"""
        conversations = []
        
        for example in examples:
            # Format full Nyaya solution
            nyaya_solution = self._format_nyaya_solution(example)
            
            # Create conversation format
            conversation = {
                "conversations": [
                    {
                        "role": "system",
                        "content": self.SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": example.problem
                    },
                    {
                        "role": "assistant",
                        "content": nyaya_solution
                    }
                ]
            }
            conversations.append(conversation)
        
        return Dataset.from_list(conversations)
    
    def _format_nyaya_solution(self, example: NyayaExample) -> str:
        """Format NyayaExample into markdown solution text"""
        solution_parts = [
            f"## Samshaya (Doubt Analysis)\n\n{example.samshaya}\n",
            f"## Pramana (Evidence Sources)\n\n{self._format_pramana(example.pramana)}\n",
            f"## Pancha Avayava (Systematic Reasoning)\n\n{self._format_pancha_avayava(example.pancha_avayava)}\n",
            f"## Tarka (Counterfactual Testing)\n\n{example.tarka}\n",
            f"## Hetvabhasa (Fallacy Detection)\n\n{self._format_hetvabhasa(example.hetvabhasa)}\n",
            f"## Nirnaya (Definitive Conclusion)\n\n{example.nirnaya}\n"
        ]
        return "\n".join(solution_parts)
    
    def _format_pramana(self, pramana: Dict[str, Any]) -> str:
        """Format Pramana section"""
        parts = []
        for source_type in ["pratyaksha", "anumana", "upamana", "shabda"]:
            if source_type in pramana:
                parts.append(f"### {source_type.capitalize()}\n\n{pramana[source_type]}\n")
        return "\n".join(parts)
    
    def _format_pancha_avayava(self, syllogisms: List[Dict[str, Any]]) -> str:
        """Format Pancha Avayava section"""
        parts = []
        for i, syllogism in enumerate(syllogisms, 1):
            parts.append(f"### Syllogism {i}\n\n")
            for component in ["pratijna", "hetu", "udaharana", "upanaya", "nigamana"]:
                if component in syllogism:
                    parts.append(f"**{component.capitalize()}**: {syllogism[component]}\n\n")
        return "\n".join(parts)
    
    def _format_hetvabhasa(self, hetvabhasa: Dict[str, Any]) -> str:
        """Format Hetvabhasa section"""
        import yaml
        return f"```yaml\n{yaml.dump(hetvabhasa, default_flow_style=False)}\n```"
```

**Data Version Manager** (`pramana/data/versioning.py`):
```python
# pramana/data/versioning.py

from pathlib import Path
from typing import Dict, Any
import yaml
from datetime import datetime

class DataVersionManager:
    """Manages data versioning and metadata"""
    
    VERSION_FILE = ".dataversion"
    
    def create_version(
        self,
        version_dir: Path,
        stage: int,
        examples_count: int,
        metadata: Dict[str, Any]
    ):
        """Create .dataversion file for a data version"""
        version_file = version_dir / self.VERSION_FILE
        
        version_data = {
            "version": self._get_version_string(version_dir),
            "created": datetime.now().isoformat(),
            "stage": stage,
            "examples_count": examples_count,
            "quality_scores": metadata.get("quality_scores", {}),
            "git_commit": self._get_git_commit(),
            "changes": metadata.get("changes", [])
        }
        
        with open(version_file, 'w') as f:
            yaml.dump(version_data, f, default_flow_style=False)
    
    def _get_version_string(self, version_dir: Path) -> str:
        """Extract version from directory name (e.g., v1.0)"""
        return version_dir.name
    
    def _get_git_commit(self) -> str:
        """Get current git commit hash"""
        import subprocess
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True
            )
            return result.stdout.strip()
        except:
            return "unknown"
```

**Usage Pattern**:
```python
# scripts/data/parse_markdown.py

from pramana.data import MarkdownParser, NyayaStructureValidator, MarkdownToTrainingConverter
from pathlib import Path

# Parse all markdown files
parser = MarkdownParser()
validator = NyayaStructureValidator()
converter = MarkdownToTrainingConverter()

examples = []
for md_file in Path("data/seed_examples/stage_zero/v1.0").glob("*.md"):
    example = parser.parse(md_file)
    result = validator.validate(example)
    
    if not result.is_valid:
        print(f"ERROR in {md_file}: {result.errors}")
        continue
    
    if result.warnings:
        print(f"WARNINGS in {md_file}: {result.warnings}")
    
    examples.append(example)

# Convert to training format
dataset = converter.convert(examples)
dataset.save_to_disk("data/processed/stage_zero_v1.0")
```

### Key Design Decisions

**1. Parser → Validator → Converter Pipeline**
- **Rationale**: Separation of concerns, testable components, reusable validators
- **Benefit**: Can validate without converting, convert without parsing

**2. Structured Data Classes**
- **Pattern**: `NyayaExample` dataclass
- **Rationale**: Type safety, IDE autocomplete, clear API
- **Benefit**: Easier to work with than raw dicts

**3. Validation Gates**
- **Pattern**: Fail fast on structure errors, warn on content issues
- **Rationale**: Catch errors before training starts
- **Implementation**: `ValidationResult` with errors/warnings separation

**4. Versioning at Directory Level**
- **Pattern**: `data/seed_examples/stage_zero/v1.0/`
- **Rationale**: Immutable versions enable reproducibility
- **Implementation**: `.dataversion` file tracks metadata

---

## 4. Evaluation Framework Pattern

### Recommendation: **Three-Tier Evaluation with Stage-Aware Metrics**

**Rationale**: Evaluation needs to evolve with project maturity:
- **Stage 0**: Format-first (structural validity)
- **Stage 1**: Structure + correctness correlation
- **Stage 2**: Custom Nyaya metrics + benchmarks
- **Stage 3**: Reward-aligned evaluation

### Implementation Pattern

**Tier 1: Structural Validator** (`pramana/evaluation/tier1_validator.py`):
```python
# pramana/evaluation/tier1_validator.py

from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
import json

@dataclass
class Tier1Result:
    """Tier 1 automated validation result"""
    passed: bool
    filters: Dict[str, bool]
    errors: List[str]

class Tier1StructuralValidator:
    """Fast automated structural validation"""
    
    def validate(self, example: Dict[str, Any]) -> Tier1Result:
        """Run all Tier 1 filters"""
        filters = {}
        errors = []
        
        # JSON schema validation
        filters["valid_json"] = self._validate_json_schema(example)
        if not filters["valid_json"]:
            errors.append("Invalid JSON schema")
        
        # Phase presence
        filters["has_all_phases"] = self._check_phases_present(example)
        if not filters["has_all_phases"]:
            errors.append("Missing required Nyaya phases")
        
        # Phase order
        filters["correct_phase_order"] = self._check_phase_order(example)
        if not filters["correct_phase_order"]:
            errors.append("Phases in incorrect order")
        
        # Component completeness
        filters["pramana_complete"] = self._check_pramana_complete(example)
        filters["avayava_complete"] = self._check_avayava_complete(example)
        filters["hetvabhasa_complete"] = self._check_hetvabhasa_complete(example)
        
        # Z3 verification (if applicable)
        if self._is_formal_logic(example):
            filters["z3_verifiable"] = self._verify_with_z3(example)
        else:
            filters["z3_verifiable"] = True  # N/A
        
        # Answer existence
        filters["has_answer"] = "answer" in example.get("nirnaya", {})
        
        # Length sanity
        example_str = json.dumps(example)
        filters["reasonable_length"] = 100 < len(example_str) < 10000
        
        # No contradictions
        filters["no_self_contradiction"] = not self._has_contradiction(example)
        
        passed = all(filters.values())
        
        return Tier1Result(passed=passed, filters=filters, errors=errors)
    
    def _validate_json_schema(self, example: Dict) -> bool:
        """Validate against JSON schema"""
        # Implementation: Use jsonschema library
        return True  # Placeholder
    
    def _check_phases_present(self, example: Dict) -> bool:
        """Check all 6 phases present"""
        required = ["samshaya", "pramana", "pancha_avayava", "tarka", "hetvabhasa", "nirnaya"]
        return all(phase in example for phase in required)
    
    def _check_phase_order(self, example: Dict) -> bool:
        """Check phases in correct order"""
        # Implementation: Check order in text/JSON structure
        return True  # Placeholder
    
    def _check_pramana_complete(self, example: Dict) -> bool:
        """Check all 4 Pramana types present"""
        pramana = example.get("pramana", {})
        required = ["pratyaksha", "anumana", "upamana", "shabda"]
        return all(source in pramana for source in required)
    
    def _check_avayava_complete(self, example: Dict) -> bool:
        """Check Pancha Avayava has all 5 components"""
        avayava = example.get("pancha_avayava", [])
        if len(avayava) == 0:
            return False
        required = ["pratijna", "hetu", "udaharana", "upanaya", "nigamana"]
        return all(comp in syllogism for syllogism in avayava for comp in required)
    
    def _check_hetvabhasa_complete(self, example: Dict) -> bool:
        """Check all 5 Hetvabhasa types checked"""
        hetvabhasa = example.get("hetvabhasa", {})
        checks = hetvabhasa.get("fallacy_checks", {})
        required = ["savyabhichara", "viruddha", "prakaranasama", "sadhyasama", "kalaatita"]
        return all(fallacy in checks for fallacy in required)
    
    def _is_formal_logic(self, example: Dict) -> bool:
        """Check if example is formal logic (CSP/SAT)"""
        problem_type = example.get("problem_type", "")
        return problem_type in ["constraint_satisfaction", "boolean_sat"]
    
    def _verify_with_z3(self, example: Dict) -> bool:
        """Verify with Z3 solver"""
        # Implementation: Use pramana.verification.z3_solver
        return True  # Placeholder
    
    def _has_contradiction(self, example: Dict) -> bool:
        """Check for internal contradictions"""
        # Implementation: Basic contradiction detection
        return False  # Placeholder
```

**Tier 2: LLM Judge** (`pramana/evaluation/tier2_judge.py`):
```python
# pramana/evaluation/tier2_judge.py

from typing import Dict, Any, Tuple
from dataclasses import dataclass
import json
import openai  # Or use your LLM client

@dataclass
class Tier2Result:
    """Tier 2 LLM judge evaluation result"""
    decision: str  # "ACCEPT", "MANUAL_REVIEW", "REJECT"
    scores: Dict[str, float]
    total_score: float
    issues: List[str]

NYAYA_EVALUATION_PROMPT = """You are an expert in Nyaya Darshan evaluating whether an AI model correctly applied the six-phase Nyaya methodology.

PROBLEM:
{problem}

PROPOSED NYAYA SOLUTION:
{solution}

Evaluate on 0-10 scale for each criterion. Provide JSON response with scores and recommendation.
[Full prompt from spec.md section 5.2]"""

class Tier2LLMJudge:
    """LLM-as-judge evaluation with Nyaya rubrics"""
    
    def __init__(self, model: str = "gpt-4-turbo-preview", temperature: float = 0.2):
        self.model = model
        self.temperature = temperature
    
    def evaluate(self, example: Dict[str, Any]) -> Tier2Result:
        """Evaluate example using LLM judge"""
        prompt = NYAYA_EVALUATION_PROMPT.format(
            problem=example.get("problem", ""),
            solution=json.dumps(example.get("nyaya_solution", {}), indent=2)
        )
        
        # Call LLM
        response = self._call_llm(prompt)
        scores = self._parse_response(response)
        
        # Decision thresholds
        total_score = scores.get("total", 0.0)
        if total_score >= 0.85:
            decision = "ACCEPT"
        elif total_score >= 0.70:
            decision = "MANUAL_REVIEW"
        else:
            decision = "REJECT"
        
        return Tier2Result(
            decision=decision,
            scores=scores,
            total_score=total_score,
            issues=scores.get("issues", [])
        )
    
    def _call_llm(self, prompt: str) -> str:
        """Call LLM API"""
        # Implementation: Use OpenAI/Anthropic API
        # For now, placeholder
        return '{"total": 0.75, "samshaya": 8, ...}'
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM JSON response"""
        try:
            return json.loads(response)
        except:
            return {"total": 0.0, "issues": ["Failed to parse LLM response"]}
```

**Tier 3: Manual Reviewer** (`pramana/evaluation/tier3_reviewer.py`):
```python
# pramana/evaluation/tier3_reviewer.py

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import random

@dataclass
class Tier3ReviewQueue:
    """Strategic manual review queue"""
    boundary_cases: List[Dict[str, Any]]
    high_scores: List[Dict[str, Any]]
    phase_failures: List[Dict[str, Any]]
    type_coverage: List[Dict[str, Any]]

class Tier3ManualReviewer:
    """Selects examples for strategic manual review"""
    
    def select_for_review(
        self,
        examples_with_scores: List[Tuple[Dict[str, Any], Dict[str, Any]]],
        max_boundary: int = 10,
        max_high: int = 15,
        max_per_phase: int = 5,
        max_per_type: int = 3
    ) -> Tier3ReviewQueue:
        """Select examples for manual review"""
        
        # Boundary cases (0.68-0.72 score range)
        boundary_cases = [
            ex for ex, scores in examples_with_scores
            if 0.68 <= scores.get("total", 0.0) <= 0.72
        ]
        boundary_cases = random.sample(boundary_cases, min(max_boundary, len(boundary_cases)))
        
        # High-scoring validation (>0.85)
        high_scores = [
            ex for ex, scores in examples_with_scores
            if scores.get("total", 0.0) >= 0.85
        ]
        high_scores = random.sample(high_scores, min(max_high, len(high_scores)))
        
        # Phase-specific failures
        phase_failures = []
        for phase in ["pratyaksha", "anumana", "tarka", "hetvabhasa"]:
            failures = [
                ex for ex, scores in examples_with_scores
                if scores.get(phase, 10) <= 5
            ]
            phase_failures.extend(random.sample(failures, min(max_per_phase, len(failures))))
        
        # Problem type coverage
        type_coverage = []
        problem_types = set(ex.get("problem_type") for ex, _ in examples_with_scores)
        for ptype in problem_types:
            type_examples = [
                ex for ex, _ in examples_with_scores
                if ex.get("problem_type") == ptype
            ]
            type_coverage.extend(random.sample(type_examples, min(max_per_type, len(type_examples))))
        
        return Tier3ReviewQueue(
            boundary_cases=boundary_cases,
            high_scores=high_scores,
            phase_failures=phase_failures,
            type_coverage=type_coverage
        )
```

**Stage-Aware Metrics** (`pramana/evaluation/metrics.py`):
```python
# pramana/evaluation/metrics.py

from typing import Dict, Any, List
from dataclasses import dataclass

@dataclass
class FormatAdherenceMetrics:
    """Format adherence metrics (Stage 0-1)"""
    has_all_phases_rate: float
    correct_order_rate: float
    phase_completeness: Dict[str, float]

@dataclass
class StructureAccuracyCorrelation:
    """Structure-accuracy correlation (Stage 1)"""
    accuracy_with_full_structure: float
    accuracy_without_full_structure: float
    correlation_coefficient: float

@dataclass
class NyayaQualityMetrics:
    """Custom Nyaya metrics (Stage 2+)"""
    pramana_validity_score: float
    z3_consistency_rate: float
    hetvabhasa_detection_f1: float
    udaharana_universal_rule_rate: float

class NyayaMetrics:
    """Stage-aware metric computation"""
    
    def compute_stage_0_metrics(self, results: List[Dict]) -> FormatAdherenceMetrics:
        """Compute Stage 0 format-first metrics"""
        total = len(results)
        has_all_phases = sum(1 for r in results if r.get("has_all_phases", False))
        correct_order = sum(1 for r in results if r.get("correct_order", False))
        
        return FormatAdherenceMetrics(
            has_all_phases_rate=has_all_phases / total,
            correct_order_rate=correct_order / total,
            phase_completeness={}  # Detailed phase breakdown
        )
    
    def compute_stage_1_correlation(self, results: List[Dict]) -> StructureAccuracyCorrelation:
        """Compute structure-accuracy correlation"""
        with_structure = [r for r in results if r.get("has_all_phases", False)]
        without_structure = [r for r in results if not r.get("has_all_phases", False)]
        
        acc_with = sum(1 for r in with_structure if r.get("correct", False)) / len(with_structure)
        acc_without = sum(1 for r in without_structure if r.get("correct", False)) / len(without_structure)
        
        # Correlation coefficient
        import numpy as np
        structure_scores = [1.0 if r.get("has_all_phases") else 0.0 for r in results]
        accuracy_scores = [1.0 if r.get("correct") else 0.0 for r in results]
        correlation = np.corrcoef(structure_scores, accuracy_scores)[0, 1]
        
        return StructureAccuracyCorrelation(
            accuracy_with_full_structure=acc_with,
            accuracy_without_full_structure=acc_without,
            correlation_coefficient=correlation
        )
    
    def compute_stage_2_nyaya_metrics(self, results: List[Dict]) -> NyayaQualityMetrics:
        """Compute custom Nyaya quality metrics"""
        # Implementation: Compute Pramana validity, Z3 consistency, etc.
        return NyayaQualityMetrics(
            pramana_validity_score=0.0,
            z3_consistency_rate=0.0,
            hetvabhasa_detection_f1=0.0,
            udaharana_universal_rule_rate=0.0
        )
```

**Usage Pattern**:
```python
# scripts/evaluation/run_tier1.py

from pramana.evaluation import Tier1StructuralValidator
from pathlib import Path
import json

validator = Tier1StructuralValidator()

for example_file in Path("data/stage_two_synthetic/batch_001").glob("*.json"):
    with open(example_file) as f:
        example = json.load(f)
    
    result = validator.validate(example)
    
    if result.passed:
        # Move to tier1_passed directory
        pass
    else:
        # Log errors
        print(f"FAILED: {example_file} - {result.errors}")
```

### Key Design Decisions

**1. Three-Tier Architecture**
- **Rationale**: Balance automation (Tier 1), scale (Tier 2), quality (Tier 3)
- **Benefit**: 85-90% quality at 20% manual effort

**2. Stage-Aware Metrics**
- **Pattern**: Different metrics per stage (format → correlation → Nyaya quality)
- **Rationale**: Evaluation evolves with project maturity
- **Implementation**: `NyayaMetrics` class with stage-specific methods

**3. LLM Judge with Explicit Rubrics**
- **Pattern**: GPT-4 evaluates against Nyaya rubrics
- **Rationale**: Catches subtle methodology errors at scale
- **Cost**: ~$5-10 for 500 examples (acceptable)

**4. Strategic Manual Review**
- **Pattern**: Not random sampling - boundary cases, high scores, phase failures
- **Rationale**: Maximize value of limited manual time
- **Implementation**: `Tier3ManualReviewer.select_for_review()`

---

## 5. Training Abstraction Level

### Recommendation: **Moderate Abstraction with Stage-Specific Trainers**

**Rationale**: 4.5-month timeline requires:
- **Speed**: Don't over-engineer, get to training quickly
- **Flexibility**: Stage-specific needs (SFT vs GRPO)
- **Reproducibility**: Save configs, seeds, checkpoints
- **Maintainability**: Clear abstractions, not too many layers

### Implementation Pattern

**Base Trainer** (`pramana/training/base_trainer.py`):
```python
# pramana/training/base_trainer.py

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any
from pramana.config import ExperimentConfig

class BaseTrainer(ABC):
    """Base trainer interface"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.output_dir = Path(config.training.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    @abstractmethod
    def prepare_model(self):
        """Load and prepare model"""
        pass
    
    @abstractmethod
    def prepare_data(self):
        """Load and prepare training data"""
        pass
    
    @abstractmethod
    def train(self):
        """Execute training"""
        pass
    
    def save_config(self, checkpoint_dir: Path):
        """Save config for reproducibility"""
        config_path = checkpoint_dir / "experiment_config.yaml"
        self.config.save(config_path)
        
        # Save git commit
        import subprocess
        try:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
            with open(checkpoint_dir / "git_commit.txt", 'w') as f:
                f.write(commit)
        except:
            pass
```

**SFT Trainer** (`pramana/training/sft_trainer.py`):
```python
# pramana/training/sft_trainer.py

from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import load_from_disk
from transformers import TrainingArguments, Trainer
from pramana.training.base_trainer import BaseTrainer
from pramana.config import ExperimentConfig
import torch

class SupervisedFineTuningTrainer(BaseTrainer):
    """Supervised fine-tuning trainer using Unsloth"""
    
    def prepare_model(self):
        """Load model with LoRA"""
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.model.base_model,
            max_seq_length=self.config.model.max_seq_length,
            dtype=None,
            load_in_4bit=self.config.model.load_in_4bit,
        )
        
        # Add LoRA adapters
        model = FastLanguageModel.get_peft_model(
            model,
            r=self.config.lora.r,
            target_modules=self.config.lora.target_modules,
            lora_alpha=self.config.lora.lora_alpha,
            lora_dropout=self.config.lora.lora_dropout,
            bias=self.config.lora.bias,
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )
        
        self.model = model
        self.tokenizer = tokenizer
    
    def prepare_data(self):
        """Load training dataset"""
        # Load from processed data directory
        dataset_path = Path(self.config.data.seed_examples_path).parent.parent / "processed" / f"stage_{self.config.experiment['stage']}_v1.0"
        self.dataset = load_from_disk(str(dataset_path))
        
        # Train/test split
        split = self.dataset.train_test_split(test_size=1 - self.config.data.train_test_split)
        self.train_dataset = split["train"]
        self.eval_dataset = split["test"]
    
    def train(self):
        """Execute training"""
        # Prepare model and data
        self.prepare_model()
        self.prepare_data()
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=self.config.training.num_train_epochs,
            per_device_train_batch_size=self.config.training.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            learning_rate=self.config.training.learning_rate,
            warmup_ratio=self.config.training.warmup_ratio,
            lr_scheduler_type=self.config.training.lr_scheduler_type,
            weight_decay=self.config.training.weight_decay,
            max_grad_norm=self.config.training.max_grad_norm,
            logging_steps=self.config.training.logging_steps,
            save_steps=self.config.training.save_steps,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            optim="adamw_torch",
            report_to="wandb" if self.config.experiment_tracking.get("wandb_project") else None,
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
        )
        
        # Train
        trainer.train()
        
        # Save final checkpoint
        final_dir = self.output_dir / "final"
        final_dir.mkdir(exist_ok=True)
        trainer.save_model(str(final_dir))
        self.tokenizer.save_pretrained(str(final_dir))
        self.save_config(final_dir)
```

**GRPO Trainer** (`pramana/training/grpo_trainer.py`):
```python
# pramana/training/grpo_trainer.py

from pramana.training.base_trainer import BaseTrainer
from pramana.config import ExperimentConfig
from pramana.evaluation import Tier1StructuralValidator, Tier2LLMJudge
from pramana.verification import Z3Solver
import torch

class GRPOTrainer(BaseTrainer):
    """GRPO trainer with custom Nyaya reward functions"""
    
    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
        self.reward_functions = self._setup_reward_functions()
    
    def _setup_reward_functions(self):
        """Setup Nyaya-specific reward functions"""
        return {
            "structure": self._compute_structure_reward,
            "z3_consistency": self._compute_z3_reward,
            "hetvabhasa": self._compute_hetvabhasa_reward,
            "pramana": self._compute_pramana_reward,
            "answer": self._compute_answer_reward,
        }
    
    def prepare_model(self):
        """Load model for GRPO"""
        # Similar to SFT but with use_cache=False for GRPO
        pass
    
    def prepare_data(self):
        """Load training prompts"""
        # Load problem statements (not full solutions)
        pass
    
    def train(self):
        """Execute GRPO training"""
        # Implementation: GRPO training loop
        # For each batch:
        #   1. Generate multiple responses per prompt
        #   2. Compute composite reward for each response
        #   3. Update policy using GRPO objective
        pass
    
    def _compute_structure_reward(self, response: str) -> float:
        """Reward structure completeness (30% weight)"""
        validator = Tier1StructuralValidator()
        result = validator.validate(self._parse_response(response))
        return 0.30 * (1.0 if result.passed else 0.0)
    
    def _compute_z3_reward(self, response: str, problem: Dict) -> float:
        """Reward Z3 consistency (25% weight)"""
        if not self._is_formal_logic(problem):
            return 0.0
        
        solver = Z3Solver()
        is_consistent = solver.verify(response, problem)
        return 0.25 * (1.0 if is_consistent else -0.5)
    
    def _compute_hetvabhasa_reward(self, response: str) -> float:
        """Reward Hetvabhasa detection (20% weight)"""
        # Implementation: Check if all 5 fallacy types checked
        return 0.20 * 0.8  # Placeholder
    
    def _compute_pramana_reward(self, response: str) -> float:
        """Reward Pramana appropriateness (15% weight)"""
        # Implementation: Validate Pramana usage
        return 0.15 * 0.9  # Placeholder
    
    def _compute_answer_reward(self, response: str, ground_truth: str) -> float:
        """Reward answer correctness (10% weight)"""
        answer = self._extract_answer(response)
        return 0.10 * (1.0 if answer == ground_truth else 0.0)
```

**Training Scripts** (`scripts/training/train_stage_zero.py`):
```python
#!/usr/bin/env python3
# scripts/training/train_stage_zero.py

import argparse
from pathlib import Path
from pramana.config import StageConfigLoader
from pramana.training import SupervisedFineTuningTrainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default="configs/stage_zero.yaml")
    parser.add_argument("--experiment-name", type=str, default="stage0-attempt01")
    parser.add_argument("--wandb-project", type=str, default="pramana-stage-zero")
    args = parser.parse_args()
    
    # Load config
    config = StageConfigLoader.load_stage_config(stage=0)
    
    # Override experiment name
    config.experiment["name"] = args.experiment_name
    config.experiment_tracking["wandb_project"] = args.wandb_project
    
    # Create trainer and train
    trainer = SupervisedFineTuningTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()
```

### Key Design Decisions

**1. Moderate Abstraction**
- **Pattern**: Base class + stage-specific implementations
- **Rationale**: Balance flexibility and speed
- **Benefit**: Can add new training methods without refactoring

**2. Unsloth Integration**
- **Pattern**: Direct use of Unsloth APIs in trainers
- **Rationale**: Don't abstract what's already simple
- **Benefit**: Leverage Unsloth optimizations directly

**3. Config-Driven Training**
- **Pattern**: All hyperparameters in config, trainers read config
- **Rationale**: Reproducibility, easy experimentation
- **Benefit**: Change configs without code changes

**4. Stage-Specific Trainers**
- **Pattern**: `SupervisedFineTuningTrainer` vs `GRPOTrainer`
- **Rationale**: Different training loops, different needs
- **Benefit**: No over-engineering, clear separation

**5. Reproducibility Built-In**
- **Pattern**: Save config + git commit with each checkpoint
- **Rationale**: Essential for research reproducibility
- **Implementation**: `BaseTrainer.save_config()`

---

## Summary & Implementation Priority

### Phase 1: Foundation (Week 1-2)
1. ✅ Project structure (`pramana/` package)
2. ✅ Config system (`pramana/config/`)
3. ✅ Data parsers (`pramana/data/parsers.py`)
4. ✅ Basic validators (`pramana/data/validators.py`)

### Phase 2: Training Infrastructure (Week 2-3)
5. ✅ SFT trainer (`pramana/training/sft_trainer.py`)
6. ✅ Training scripts (`scripts/training/`)
7. ✅ Experiment tracking integration

### Phase 3: Evaluation (Week 4-5)
8. ✅ Tier 1 validator (`pramana/evaluation/tier1_validator.py`)
9. ✅ Tier 2 LLM judge (`pramana/evaluation/tier2_judge.py`)
10. ✅ Metrics computation (`pramana/evaluation/metrics.py`)

### Phase 4: Advanced Features (Week 6+)
11. ✅ Z3 integration (`pramana/verification/`)
12. ✅ GRPO trainer (`pramana/training/grpo_trainer.py`)
13. ✅ Tier 3 manual review (`pramana/evaluation/tier3_reviewer.py`)

---

## Critical Success Factors

1. **Start Simple**: Implement Phase 1 first, validate with Stage 0
2. **Iterate Based on Needs**: Add abstractions only when needed
3. **Reproducibility First**: Always save configs, seeds, versions
4. **Test Early**: Unit tests for parsers, validators before training
5. **Document Decisions**: Update this doc as architecture evolves

---

**Next Steps**: Implement Phase 1 components, validate with Stage 0 data, then proceed to Phase 2.
