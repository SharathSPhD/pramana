# ML Training Pipeline Design Patterns

## Architecture Blueprint for Pramana Staged Training

This document provides comprehensive design patterns for implementing the Pramana ML training pipeline with Unsloth + QLoRA, staged training, GRPO, custom rewards, and multi-tier evaluation.

---

## Patterns & Conventions Found

### Existing Patterns
- **Staged Training**: 4-stage progression (POC → MVP → Scaling → RL) with validation gates
- **Configuration-Driven**: YAML configs per stage (`stage_zero_config.yaml`, `grpo_stage_three_config.yaml`)
- **Experiment Tracking**: Weights & Biases / TensorBoard mentioned in CLAUDE.md
- **Data Format**: Structured Markdown with YAML frontmatter (human-readable, Git-friendly)
- **Reward Composition**: 5-component weighted reward function (30% structure, 25% Z3, 20% Hetvabhasa, 15% Pramana, 10% answer)
- **Evaluation Tiers**: Tier 1 (automated) → Tier 2 (LLM judge) → Tier 3 (manual review)

### Key Constraints
- Unsloth + QLoRA for efficient fine-tuning (4-bit quantization)
- GRPO (Group Relative Policy Optimization) for RL stage
- Z3 SMT solver integration for formal logic verification
- Immutable data versioning (Git + `.dataversion` files)
- Checkpoint metadata must include git commit, epoch, loss, timestamp

---

## Architecture Decisions

### 1. Trainer Abstraction Pattern

**Decision**: Template Method Pattern with Strategy for stage-specific behavior

**Rationale**:
- Stages 0-2 use supervised fine-tuning (SFT) with Unsloth
- Stage 3 uses GRPO (reinforcement learning)
- Common setup/teardown/logging logic shared via base class
- Stage-specific training loops isolated in subclasses

**Trade-offs**:
- ✅ Clear separation of concerns
- ✅ Easy to add new stages
- ✅ Shared infrastructure (checkpointing, logging)
- ⚠️ Slight overhead from abstraction (negligible)

### 2. Reward Function Pattern

**Decision**: Composite Pattern with Weighted Strategy Components

**Rationale**:
- 5 independent reward components with different weights
- Some rewards are optional (Z3 only for formalizable problems)
- Need to track individual component scores for debugging
- Reward functions may evolve independently

**Trade-offs**:
- ✅ Modular, testable components
- ✅ Easy to adjust weights without code changes
- ✅ Can disable/enable components per problem type
- ⚠️ Requires careful normalization to prevent component dominance

### 3. Evaluation Pipeline Pattern

**Decision**: Chain of Responsibility with Early Termination

**Rationale**:
- Tier 1 (automated) filters out obvious failures
- Tier 2 (LLM judge) only runs on Tier 1 passes (cost optimization)
- Tier 3 (manual) only for gold-standard examples
- Each tier can short-circuit to next stage

**Trade-offs**:
- ✅ Cost-efficient (don't run expensive LLM judge on failures)
- ✅ Clear quality gates
- ✅ Parallelizable within tiers
- ⚠️ Tier dependencies require careful ordering

### 4. Checkpoint Management Pattern

**Decision**: Repository Pattern with Metadata Enrichment

**Rationale**:
- Checkpoints need rich metadata for experiment tracking
- Must link to data versions, git commits, hyperparameters
- Support checkpoint comparison and rollback
- Integration with W&B/TensorBoard

**Trade-offs**:
- ✅ Full experiment reproducibility
- ✅ Easy checkpoint comparison
- ✅ Supports A/B testing workflows
- ⚠️ Metadata storage overhead (minimal)

### 5. Data Versioning Pattern

**Decision**: Immutable Snapshots with Git + Content-Addressable Storage

**Rationale**:
- Training data must be immutable once used
- Need to track quality scores and changes over time
- Git provides audit trail
- `.dataversion` files provide machine-readable metadata

**Trade-offs**:
- ✅ Full reproducibility
- ✅ Clear data lineage
- ✅ Easy to rollback to previous versions
- ⚠️ Storage overhead (acceptable for research project)

---

## Component Design

### 1. Trainer Abstraction Pattern

#### BaseTrainer (Abstract Base Class)

**File**: `pramana/training/base_trainer.py`

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional
import torch
from unsloth import FastLanguageModel
from transformers import TrainingArguments
import wandb

@dataclass
class TrainingStage:
    """Enum-like class for training stages"""
    POC = 0
    MVP = 1
    SCALING = 2
    RL = 3

@dataclass
class TrainerConfig:
    """Base configuration for all trainers"""
    stage: TrainingStage
    experiment_name: str
    output_dir: Path
    base_model: str
    max_seq_length: int = 4096
    load_in_4bit: bool = True
    lora_config: Optional[Dict[str, Any]] = None
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    seed: int = 3407

class BaseTrainer(ABC):
    """
    Abstract base trainer implementing Template Method pattern.
    
    Subclasses implement stage-specific training logic while
    inheriting common setup, checkpointing, and logging.
    """
    
    def __init__(self, config: TrainerConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.training_args = None
        self.checkpoint_manager = None
        
    def train(self) -> Dict[str, Any]:
        """
        Template method defining the training workflow.
        Subclasses override _build_trainer() and _run_training().
        """
        # Setup phase (common to all stages)
        self._setup_logging()
        self._load_model()
        self._prepare_data()
        self._build_training_args()
        
        # Stage-specific training
        trainer = self._build_trainer()
        results = self._run_training(trainer)
        
        # Teardown phase (common to all stages)
        self._save_final_checkpoint()
        self._log_final_metrics(results)
        
        return results
    
    def _setup_logging(self):
        """Initialize W&B or TensorBoard logging"""
        if self.config.wandb_project:
            wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_config.wandb_entity,
                name=self.config.experiment_name,
                config=self.config.__dict__
            )
    
    def _load_model(self):
        """Load base model with Unsloth optimizations"""
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.base_model,
            max_seq_length=self.config.max_seq_length,
            dtype=None,
            load_in_4bit=self.config.load_in_4bit,
        )
        
        # Apply LoRA if configured
        if self.config.lora_config:
            self.model = FastLanguageModel.get_peft_model(
                self.model,
                **self.config.lora_config
            )
    
    @abstractmethod
    def _prepare_data(self):
        """Load and preprocess training data (stage-specific)"""
        pass
    
    def _build_training_args(self):
        """Create TrainingArguments (can be overridden)"""
        self.training_args = TrainingArguments(
            output_dir=str(self.config.output_dir),
            per_device_train_batch_size=self._get_batch_size(),
            gradient_accumulation_steps=self._get_gradient_accumulation(),
            num_train_epochs=self._get_epochs(),
            learning_rate=self._get_learning_rate(),
            logging_steps=self._get_logging_steps(),
            save_steps=self._get_save_steps(),
            save_strategy="steps",
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            optim="adamw_8bit",
            warmup_steps=self._get_warmup_steps(),
            report_to="wandb" if self.config.wandb_project else "tensorboard",
            run_name=self.config.experiment_name,
            seed=self.config.seed,
        )
    
    @abstractmethod
    def _build_trainer(self):
        """Build stage-specific trainer (SFTTrainer, GRPOTrainer, etc.)"""
        pass
    
    @abstractmethod
    def _run_training(self, trainer) -> Dict[str, Any]:
        """Execute training loop (stage-specific)"""
        pass
    
    def _save_final_checkpoint(self):
        """Save final model checkpoint with metadata"""
        checkpoint_path = self.config.output_dir / "final"
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        self.model.save_pretrained(str(checkpoint_path))
        self.tokenizer.save_pretrained(str(checkpoint_path))
        
        # Save metadata
        self._save_checkpoint_metadata(checkpoint_path)
    
    def _save_checkpoint_metadata(self, checkpoint_path: Path):
        """Enrich checkpoint with experiment metadata"""
        import subprocess
        import json
        from datetime import datetime
        
        # Get git commit hash
        try:
            git_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], 
                cwd=Path(__file__).parent.parent.parent
            ).decode().strip()
        except:
            git_commit = "unknown"
        
        metadata = {
            "experiment_name": self.config.experiment_name,
            "stage": self.config.stage.value if hasattr(self.config.stage, 'value') else self.config.stage,
            "checkpoint_path": str(checkpoint_path),
            "timestamp": datetime.utcnow().isoformat(),
            "git_commit": git_commit,
            "base_model": self.config.base_model,
            "max_seq_length": self.config.max_seq_length,
            "hyperparameters": self.training_args.to_dict() if self.training_args else {},
        }
        
        with open(checkpoint_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
    
    def _log_final_metrics(self, results: Dict[str, Any]):
        """Log final metrics to W&B/TensorBoard"""
        if self.config.wandb_project:
            wandb.log(results)
            wandb.finish()
    
    # Abstract methods for stage-specific hyperparameters
    @abstractmethod
    def _get_batch_size(self) -> int:
        pass
    
    @abstractmethod
    def _get_gradient_accumulation(self) -> int:
        pass
    
    @abstractmethod
    def _get_epochs(self) -> int:
        pass
    
    @abstractmethod
    def _get_learning_rate(self) -> float:
        pass
    
    @abstractmethod
    def _get_logging_steps(self) -> int:
        pass
    
    @abstractmethod
    def _get_save_steps(self) -> int:
        pass
    
    @abstractmethod
    def _get_warmup_steps(self) -> int:
        pass
```

#### Stage-Specific Trainers

**File**: `pramana/training/sft_trainer.py` (Stages 0-2)

```python
from typing import Dict, Any
from pathlib import Path
from trl import SFTTrainer
from datasets import Dataset
from .base_trainer import BaseTrainer, TrainerConfig, TrainingStage

class SFTTrainerWrapper(BaseTrainer):
    """
    Supervised Fine-Tuning trainer for Stages 0-2.
    Uses Unsloth's SFTTrainer with QLoRA.
    """
    
    def __init__(self, config: TrainerConfig, train_dataset: Dataset, val_dataset: Dataset = None):
        super().__init__(config)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
    
    def _prepare_data(self):
        """Data already provided in constructor"""
        # Could add data validation/transformation here
        pass
    
    def _build_trainer(self) -> SFTTrainer:
        """Build Unsloth SFTTrainer"""
        return SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            dataset_text_field="text",  # Field containing training text
            max_seq_length=self.config.max_seq_length,
            args=self.training_args,
        )
    
    def _run_training(self, trainer: SFTTrainer) -> Dict[str, Any]:
        """Execute SFT training"""
        train_results = trainer.train()
        
        # Evaluate if validation set provided
        eval_results = {}
        if self.val_dataset:
            eval_results = trainer.evaluate()
        
        return {
            "train_loss": train_results.training_loss,
            "train_runtime": train_results.metrics.get("train_runtime", 0),
            **eval_results
        }
    
    # Stage-specific hyperparameter overrides
    def _get_batch_size(self) -> int:
        if self.config.stage == TrainingStage.POC:
            return 2  # Small for POC
        elif self.config.stage == TrainingStage.MVP:
            return 2
        else:  # SCALING
            return 4
    
    def _get_gradient_accumulation(self) -> int:
        if self.config.stage == TrainingStage.POC:
            return 4
        elif self.config.stage == TrainingStage.MVP:
            return 8
        else:  # SCALING
            return 8
    
    def _get_epochs(self) -> int:
        if self.config.stage == TrainingStage.POC:
            return 10  # Overfit expected
        elif self.config.stage == TrainingStage.MVP:
            return 10
        else:  # SCALING
            return 15
    
    def _get_learning_rate(self) -> float:
        return 2e-4  # LoRA uses higher LR
    
    def _get_logging_steps(self) -> int:
        return 10
    
    def _get_save_steps(self) -> int:
        if self.config.stage == TrainingStage.POC:
            return 50
        else:
            return 100
    
    def _get_warmup_steps(self) -> int:
        return 50
```

**File**: `pramana/training/grpo_trainer.py` (Stage 3)

```python
from typing import Dict, Any, List, Callable
from pathlib import Path
from datasets import Dataset
from trl import GRPOTrainer
from .base_trainer import BaseTrainer, TrainerConfig, TrainingStage
from ..rewards import CompositeRewardFunction

class GRPOTrainerWrapper(BaseTrainer):
    """
    GRPO (Group Relative Policy Optimization) trainer for Stage 3.
    Uses custom composite reward function.
    """
    
    def __init__(
        self, 
        config: TrainerConfig,
        train_dataset: Dataset,
        reward_function: CompositeRewardFunction,
        num_generations: int = 4,
        beta: float = 0.01,
    ):
        super().__init__(config)
        self.train_dataset = train_dataset
        self.reward_function = reward_function
        self.num_generations = num_generations
        self.beta = beta
    
    def _prepare_data(self):
        """Prepare dataset for GRPO (needs prompt formatting)"""
        # GRPO needs prompts, not full conversations
        pass
    
    def _build_trainer(self) -> GRPOTrainer:
        """Build GRPO trainer with custom reward"""
        return GRPOTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=self.training_args,
            train_dataset=self.train_dataset,
            reward_function=self.reward_function.compute,
            num_generations=self.num_generations,
            beta=self.beta,
        )
    
    def _run_training(self, trainer: GRPOTrainer) -> Dict[str, Any]:
        """Execute GRPO training"""
        train_results = trainer.train()
        
        return {
            "train_loss": train_results.training_loss,
            "train_runtime": train_results.metrics.get("train_runtime", 0),
            "mean_reward": train_results.metrics.get("mean_reward", 0),
        }
    
    # GRPO-specific hyperparameters
    def _get_batch_size(self) -> int:
        return 2
    
    def _get_gradient_accumulation(self) -> int:
        return 8
    
    def _get_epochs(self) -> int:
        return 1  # GRPO uses iterations, not epochs
    
    def _get_learning_rate(self) -> float:
        return 5e-6  # Lower LR for RL
    
    def _get_logging_steps(self) -> int:
        return 5
    
    def _get_save_steps(self) -> int:
        return 50
    
    def _get_warmup_steps(self) -> int:
        return 50
```

---

### 2. Reward Function Pattern

**File**: `pramana/rewards/composite_reward.py`

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import numpy as np

@dataclass
class RewardComponent:
    """Individual reward component with weight"""
    name: str
    weight: float
    compute_fn: callable
    enabled: bool = True
    normalize: bool = True  # Whether to normalize to [0, 1]
    
    def compute(self, *args, **kwargs) -> float:
        """Compute reward component"""
        if not self.enabled:
            return 0.0
        
        raw_score = self.compute_fn(*args, **kwargs)
        
        # Normalize if needed (assumes compute_fn returns [0, 1] or [-1, 1])
        if self.normalize and raw_score < 0:
            # Handle negative rewards (e.g., Z3 penalty)
            return raw_score
        
        return raw_score

class CompositeRewardFunction:
    """
    Composite reward function combining multiple weighted components.
    
    Uses Strategy pattern for individual components and Composite
    pattern for aggregation.
    """
    
    def __init__(self, components: List[RewardComponent]):
        self.components = components
        self._validate_weights()
    
    def _validate_weights(self):
        """Ensure weights sum to reasonable range (not necessarily 1.0)"""
        total_weight = sum(c.weight for c in self.components if c.enabled)
        if total_weight > 2.0:
            raise ValueError(f"Total weight {total_weight} exceeds 2.0")
    
    def compute(
        self, 
        generated_solution: str,
        problem: Dict[str, Any],
        ground_truth: Optional[str] = None,
        **kwargs
    ) -> tuple[float, Dict[str, float]]:
        """
        Compute composite reward.
        
        Returns:
            (total_reward, component_scores)
        """
        component_scores = {}
        
        for component in self.components:
            try:
                score = component.compute(
                    generated_solution=generated_solution,
                    problem=problem,
                    ground_truth=ground_truth,
                    **kwargs
                )
                weighted_score = component.weight * score
                component_scores[component.name] = {
                    "raw": score,
                    "weighted": weighted_score
                }
            except Exception as e:
                # Log error but don't fail entire reward computation
                print(f"Error computing {component.name} reward: {e}")
                component_scores[component.name] = {
                    "raw": 0.0,
                    "weighted": 0.0,
                    "error": str(e)
                }
        
        total_reward = sum(
            scores["weighted"] 
            for scores in component_scores.values()
        )
        
        return total_reward, component_scores
    
    def enable_component(self, name: str):
        """Enable a reward component"""
        for component in self.components:
            if component.name == name:
                component.enabled = True
                break
    
    def disable_component(self, name: str):
        """Disable a reward component"""
        for component in self.components:
            if component.name == name:
                component.enabled = False
                break
```

**File**: `pramana/rewards/nyaya_rewards.py`

```python
from typing import Dict, Any, Optional
from .composite_reward import RewardComponent, CompositeRewardFunction
from ..validation import NyayaStructureValidator
from ..validation import Z3Verifier
from ..validation import HetvabhasaDetector
from ..validation import PramanaScorer

def create_nyaya_reward_function() -> CompositeRewardFunction:
    """
    Factory function creating the standard Nyaya reward function
    with 5 weighted components.
    """
    
    structure_validator = NyayaStructureValidator()
    z3_verifier = Z3Verifier()
    hetvabhasa_detector = HetvabhasaDetector()
    pramana_scorer = PramanaScorer()
    
    components = [
        # R1: Structural completeness (30%)
        RewardComponent(
            name="structure",
            weight=0.30,
            compute_fn=lambda gen, prob, gt, **kw: (
                1.0 if structure_validator.validate(gen)["has_all_phases"] else 0.0
            ),
        ),
        
        # R2: Logical consistency via Z3 (25%)
        RewardComponent(
            name="z3_consistency",
            weight=0.25,
            compute_fn=lambda gen, prob, gt, **kw: (
                _compute_z3_reward(gen, prob, z3_verifier)
            ),
        ),
        
        # R3: Hetvabhasa detection (20%)
        RewardComponent(
            name="hetvabhasa",
            weight=0.20,
            compute_fn=lambda gen, prob, gt, **kw: (
                hetvabhasa_detector.score(gen)
            ),
        ),
        
        # R4: Pramana appropriateness (15%)
        RewardComponent(
            name="pramana",
            weight=0.15,
            compute_fn=lambda gen, prob, gt, **kw: (
                pramana_scorer.score(gen, prob)
            ),
        ),
        
        # R5: Answer correctness (10%)
        RewardComponent(
            name="answer",
            weight=0.10,
            compute_fn=lambda gen, prob, gt, **kw: (
                1.0 if _extract_answer(gen) == gt else 0.0
            ) if gt else 0.0,
        ),
    ]
    
    return CompositeRewardFunction(components)

def _compute_z3_reward(generated_solution: str, problem: Dict[str, Any], z3_verifier) -> float:
    """Compute Z3 consistency reward (only for formalizable problems)"""
    if not z3_verifier.is_formalizable(problem):
        return 0.0  # Neutral if not applicable
    
    is_valid = z3_verifier.verify(generated_solution, problem)
    if is_valid:
        return 1.0
    else:
        return -0.5  # Penalty for logical inconsistency

def _extract_answer(generated_solution: str) -> Optional[str]:
    """Extract final answer from Nirnaya phase"""
    # Implementation depends on output format
    # Assumes structured JSON or markdown parsing
    import json
    try:
        parsed = json.loads(generated_solution)
        return parsed.get("nirnaya", {}).get("answer")
    except:
        # Fallback to text parsing
        return None
```

---

### 3. Evaluation Pipeline Pattern

**File**: `pramana/evaluation/evaluation_pipeline.py`

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from enum import Enum
import asyncio
from pathlib import Path

class EvaluationTier(Enum):
    """Evaluation tier levels"""
    TIER1_AUTOMATED = 1
    TIER2_LLM_JUDGE = 2
    TIER3_MANUAL = 3

@dataclass
class EvaluationResult:
    """Result from a single evaluation tier"""
    tier: EvaluationTier
    passed: bool
    score: float
    metrics: Dict[str, Any]
    errors: List[str]
    should_continue: bool  # Whether to proceed to next tier

class EvaluationHandler(ABC):
    """
    Base handler for Chain of Responsibility pattern.
    Each tier is a handler that can pass to next tier.
    """
    
    def __init__(self, next_handler: Optional['EvaluationHandler'] = None):
        self.next_handler = next_handler
    
    def evaluate(self, example: Dict[str, Any], model_output: str) -> List[EvaluationResult]:
        """
        Evaluate and potentially chain to next handler.
        
        Returns:
            List of evaluation results from this tier and subsequent tiers
        """
        result = self._evaluate(example, model_output)
        results = [result]
        
        # Chain to next tier if this tier passed and should continue
        if result.should_continue and self.next_handler:
            next_results = self.next_handler.evaluate(example, model_output)
            results.extend(next_results)
        
        return results
    
    @abstractmethod
    def _evaluate(self, example: Dict[str, Any], model_output: str) -> EvaluationResult:
        """Perform tier-specific evaluation"""
        pass

class Tier1AutomatedHandler(EvaluationHandler):
    """
    Tier 1: Automated structural and format validation.
    Fast, deterministic, filters obvious failures.
    """
    
    def __init__(self, validator, next_handler: Optional[EvaluationHandler] = None):
        super().__init__(next_handler)
        self.validator = validator
    
    def _evaluate(self, example: Dict[str, Any], model_output: str) -> EvaluationResult:
        """Run automated validation checks"""
        validation_result = self.validator.validate(model_output)
        
        # Check format adherence
        format_score = 1.0 if validation_result["has_all_phases"] else 0.0
        
        # Check phase completeness
        phase_completeness = validation_result.get("phase_completeness", {})
        completeness_score = sum(
            1.0 for phase in phase_completeness.values() 
            if phase.get("complete", False)
        ) / len(phase_completeness) if phase_completeness else 0.0
        
        # Combined score
        score = 0.7 * format_score + 0.3 * completeness_score
        
        # Pass threshold: 0.7 (70% of structure correct)
        passed = score >= 0.7
        
        metrics = {
            "format_score": format_score,
            "completeness_score": completeness_score,
            "validation_details": validation_result
        }
        
        return EvaluationResult(
            tier=EvaluationTier.TIER1_AUTOMATED,
            passed=passed,
            score=score,
            metrics=metrics,
            errors=[] if passed else ["Format validation failed"],
            should_continue=passed  # Only continue if passed
        )

class Tier2LLMJudgeHandler(EvaluationHandler):
    """
    Tier 2: LLM-based quality assessment.
    Only runs on Tier 1 passes (cost optimization).
    """
    
    def __init__(self, llm_judge, next_handler: Optional[EvaluationHandler] = None):
        super().__init__(next_handler)
        self.llm_judge = llm_judge
    
    def _evaluate(self, example: Dict[str, Any], model_output: str) -> EvaluationResult:
        """Run LLM judge evaluation"""
        try:
            judgment = self.llm_judge.evaluate(
                problem=example["problem"],
                solution=model_output,
                ground_truth=example.get("ground_truth")
            )
            
            score = judgment["overall_score"]
            passed = score >= 0.75  # 75% threshold for Tier 2
            
            metrics = {
                "overall_score": score,
                "component_scores": judgment.get("component_scores", {}),
                "reasoning_quality": judgment.get("reasoning_quality", 0),
                "nyaya_adherence": judgment.get("nyaya_adherence", 0),
            }
            
            errors = judgment.get("errors", [])
            
            return EvaluationResult(
                tier=EvaluationTier.TIER2_LLM_JUDGE,
                passed=passed,
                score=score,
                metrics=metrics,
                errors=errors,
                should_continue=passed  # Continue to manual if passed
            )
        except Exception as e:
            return EvaluationResult(
                tier=EvaluationTier.TIER2_LLM_JUDGE,
                passed=False,
                score=0.0,
                metrics={},
                errors=[f"LLM judge error: {str(e)}"],
                should_continue=False
            )

class Tier3ManualHandler(EvaluationHandler):
    """
    Tier 3: Manual expert review.
    Only for gold-standard examples or high-stakes decisions.
    """
    
    def __init__(self, review_db_path: Optional[Path] = None):
        super().__init__(None)  # Manual is final tier
        self.review_db_path = review_db_path
    
    def _evaluate(self, example: Dict[str, Any], model_output: str) -> EvaluationResult:
        """
        Manual review (returns placeholder - actual review done offline).
        This handler queues examples for manual review.
        """
        # In practice, this would:
        # 1. Save example + output to review queue
        # 2. Notify reviewers
        # 3. Wait for manual annotation
        
        # For now, return placeholder
        return EvaluationResult(
            tier=EvaluationTier.TIER3_MANUAL,
            passed=False,  # Unknown until reviewed
            score=0.0,
            metrics={"status": "queued_for_review"},
            errors=[],
            should_continue=False  # Final tier
        )

class EvaluationPipeline:
    """
    Main evaluation pipeline orchestrating the chain of handlers.
    """
    
    def __init__(
        self,
        tier1_handler: Tier1AutomatedHandler,
        tier2_handler: Optional[Tier2LLMJudgeHandler] = None,
        tier3_handler: Optional[Tier3ManualHandler] = None,
    ):
        # Build chain: Tier1 -> Tier2 -> Tier3
        if tier3_handler:
            tier2_handler.next_handler = tier3_handler if tier2_handler else None
        if tier2_handler:
            tier1_handler.next_handler = tier2_handler
        
        self.tier1_handler = tier1_handler
    
    def evaluate(
        self, 
        example: Dict[str, Any], 
        model_output: str,
        max_tier: EvaluationTier = EvaluationTier.TIER2_LLM_JUDGE
    ) -> List[EvaluationResult]:
        """
        Run evaluation pipeline up to max_tier.
        
        Args:
            example: Problem example with ground truth
            model_output: Model-generated solution
            max_tier: Maximum tier to evaluate (cost control)
        """
        results = self.tier1_handler.evaluate(example, model_output)
        
        # Filter to max_tier
        return [r for r in results if r.tier.value <= max_tier.value]
    
    def evaluate_batch(
        self,
        examples: List[Dict[str, Any]],
        model_outputs: List[str],
        max_tier: EvaluationTier = EvaluationTier.TIER2_LLM_JUDGE,
    ) -> List[List[EvaluationResult]]:
        """Evaluate batch of examples (can parallelize)"""
        return [
            self.evaluate(ex, out, max_tier)
            for ex, out in zip(examples, model_outputs)
        ]
```

---

### 4. Checkpoint Management Pattern

**File**: `pramana/training/checkpoint_manager.py`

```python
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, Optional, List
import json
import subprocess
from datetime import datetime
import shutil

@dataclass
class CheckpointMetadata:
    """Rich metadata for a checkpoint"""
    checkpoint_id: str
    experiment_name: str
    stage: int
    epoch: Optional[int] = None
    step: Optional[int] = None
    timestamp: str = ""
    git_commit: str = ""
    git_branch: str = ""
    base_model: str = ""
    data_version: str = ""
    hyperparameters: Dict[str, Any] = None
    metrics: Dict[str, float] = None
    checkpoint_path: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()
        if self.hyperparameters is None:
            self.hyperparameters = {}
        if self.metrics is None:
            self.metrics = {}

class CheckpointRepository:
    """
    Repository pattern for checkpoint management.
    Handles storage, retrieval, comparison, and metadata enrichment.
    """
    
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir = self.base_dir / ".metadata"
        self.metadata_dir.mkdir(exist_ok=True)
    
    def save_checkpoint(
        self,
        model_path: Path,
        tokenizer_path: Path,
        metadata: CheckpointMetadata,
        copy_files: bool = True
    ) -> Path:
        """
        Save checkpoint with enriched metadata.
        
        Args:
            model_path: Path to saved model
            tokenizer_path: Path to saved tokenizer
            metadata: Checkpoint metadata
            copy_files: Whether to copy files (False = symlink)
        
        Returns:
            Path to saved checkpoint directory
        """
        # Create checkpoint directory
        checkpoint_dir = self.base_dir / metadata.checkpoint_id
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy/link model files
        if copy_files:
            shutil.copytree(model_path, checkpoint_dir / "model", dirs_exist_ok=True)
            shutil.copytree(tokenizer_path, checkpoint_dir / "tokenizer", dirs_exist_ok=True)
        else:
            # Create symlinks (saves space)
            (checkpoint_dir / "model").symlink_to(model_path)
            (checkpoint_dir / "tokenizer").symlink_to(tokenizer_path)
        
        # Enrich metadata
        enriched_metadata = self._enrich_metadata(metadata)
        enriched_metadata.checkpoint_path = str(checkpoint_dir)
        
        # Save metadata
        metadata_file = self.metadata_dir / f"{metadata.checkpoint_id}.json"
        with open(metadata_file, "w") as f:
            json.dump(asdict(enriched_metadata), f, indent=2)
        
        # Update index
        self._update_index(enriched_metadata)
        
        return checkpoint_dir
    
    def _enrich_metadata(self, metadata: CheckpointMetadata) -> CheckpointMetadata:
        """Enrich metadata with git info, data versions, etc."""
        # Get git commit
        if not metadata.git_commit:
            try:
                metadata.git_commit = subprocess.check_output(
                    ["git", "rev-parse", "HEAD"],
                    cwd=Path(__file__).parent.parent.parent
                ).decode().strip()
            except:
                metadata.git_commit = "unknown"
        
        # Get git branch
        if not metadata.git_branch:
            try:
                metadata.git_branch = subprocess.check_output(
                    ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                    cwd=Path(__file__).parent.parent.parent
                ).decode().strip()
            except:
                metadata.git_branch = "unknown"
        
        # Get data version (from .dataversion file)
        if not metadata.data_version:
            data_version_file = Path(__file__).parent.parent.parent / "data" / ".dataversion"
            if data_version_file.exists():
                with open(data_version_file) as f:
                    data_info = json.load(f)
                    metadata.data_version = data_info.get("version", "unknown")
        
        return metadata
    
    def load_checkpoint(self, checkpoint_id: str) -> tuple[Path, CheckpointMetadata]:
        """
        Load checkpoint by ID.
        
        Returns:
            (checkpoint_path, metadata)
        """
        checkpoint_dir = self.base_dir / checkpoint_id
        if not checkpoint_dir.exists():
            raise ValueError(f"Checkpoint {checkpoint_id} not found")
        
        metadata_file = self.metadata_dir / f"{checkpoint_id}.json"
        if not metadata_file.exists():
            raise ValueError(f"Metadata for {checkpoint_id} not found")
        
        with open(metadata_file) as f:
            metadata_dict = json.load(f)
            metadata = CheckpointMetadata(**metadata_dict)
        
        return checkpoint_dir, metadata
    
    def list_checkpoints(
        self,
        experiment_name: Optional[str] = None,
        stage: Optional[int] = None,
        sort_by: str = "timestamp"
    ) -> List[CheckpointMetadata]:
        """List checkpoints with optional filtering"""
        checkpoints = []
        
        for metadata_file in self.metadata_dir.glob("*.json"):
            with open(metadata_file) as f:
                metadata_dict = json.load(f)
                metadata = CheckpointMetadata(**metadata_dict)
                
                # Filter
                if experiment_name and metadata.experiment_name != experiment_name:
                    continue
                if stage is not None and metadata.stage != stage:
                    continue
                
                checkpoints.append(metadata)
        
        # Sort
        if sort_by == "timestamp":
            checkpoints.sort(key=lambda x: x.timestamp, reverse=True)
        elif sort_by == "step":
            checkpoints.sort(key=lambda x: x.step or 0, reverse=True)
        elif sort_by == "metrics":
            # Sort by primary metric (e.g., validation loss)
            checkpoints.sort(
                key=lambda x: x.metrics.get("eval_loss", float("inf")),
                reverse=False
            )
        
        return checkpoints
    
    def compare_checkpoints(
        self,
        checkpoint_id1: str,
        checkpoint_id2: str
    ) -> Dict[str, Any]:
        """Compare two checkpoints"""
        _, meta1 = self.load_checkpoint(checkpoint_id1)
        _, meta2 = self.load_checkpoint(checkpoint_id2)
        
        comparison = {
            "checkpoint1": checkpoint_id1,
            "checkpoint2": checkpoint_id2,
            "differences": {}
        }
        
        # Compare hyperparameters
        if meta1.hyperparameters != meta2.hyperparameters:
            comparison["differences"]["hyperparameters"] = {
                "checkpoint1": meta1.hyperparameters,
                "checkpoint2": meta2.hyperparameters
            }
        
        # Compare metrics
        if meta1.metrics != meta2.metrics:
            comparison["differences"]["metrics"] = {
                "checkpoint1": meta1.metrics,
                "checkpoint2": meta2.metrics
            }
        
        # Compare data versions
        if meta1.data_version != meta2.data_version:
            comparison["differences"]["data_version"] = {
                "checkpoint1": meta1.data_version,
                "checkpoint2": meta2.data_version
            }
        
        return comparison
    
    def _update_index(self, metadata: CheckpointMetadata):
        """Update checkpoint index for fast lookup"""
        index_file = self.metadata_dir / "index.json"
        
        if index_file.exists():
            with open(index_file) as f:
                index = json.load(f)
        else:
            index = {}
        
        index[metadata.checkpoint_id] = {
            "experiment_name": metadata.experiment_name,
            "stage": metadata.stage,
            "timestamp": metadata.timestamp,
            "checkpoint_path": metadata.checkpoint_path
        }
        
        with open(index_file, "w") as f:
            json.dump(index, f, indent=2)
```

---

### 5. Data Versioning Pattern

**File**: `pramana/data/versioning.py`

```python
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import hashlib
import subprocess
from datetime import datetime

@dataclass
class DataVersion:
    """Immutable data version metadata"""
    version: str  # Semantic version: "1.0", "1.1", "2.0"
    created: str  # ISO timestamp
    examples_count: int
    quality_scores: Dict[str, float]
    changes: List[str]  # Changelog entries
    git_commit: str
    checksum: str  # Content hash for verification
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class DataVersionManager:
    """
    Manages immutable data versions with Git integration.
    Uses content-addressable storage pattern.
    """
    
    def __init__(self, data_root: Path):
        self.data_root = Path(data_root)
        self.version_file = self.data_root / ".dataversion"
    
    def create_version(
        self,
        version: str,
        examples_path: Path,
        quality_scores: Dict[str, float],
        changes: List[str],
        tag_in_git: bool = True
    ) -> DataVersion:
        """
        Create a new immutable data version.
        
        Args:
            version: Semantic version string
            examples_path: Path to versioned examples
            quality_scores: Quality metrics for this version
            changes: Changelog describing what changed
            tag_in_git: Whether to create git tag
        
        Returns:
            DataVersion object
        """
        # Compute checksum of examples
        checksum = self._compute_checksum(examples_path)
        
        # Get git commit
        try:
            git_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=self.data_root.parent
            ).decode().strip()
        except:
            git_commit = "unknown"
        
        # Count examples
        examples_count = self._count_examples(examples_path)
        
        # Create version object
        data_version = DataVersion(
            version=version,
            created=datetime.utcnow().isoformat(),
            examples_count=examples_count,
            quality_scores=quality_scores,
            changes=changes,
            git_commit=git_commit,
            checksum=checksum
        )
        
        # Save version file
        self._save_version(data_version)
        
        # Create git tag if requested
        if tag_in_git:
            try:
                tag_name = f"data-v{version}"
                subprocess.run(
                    ["git", "tag", tag_name, "-m", f"Data version {version}"],
                    cwd=self.data_root.parent,
                    check=True
                )
            except Exception as e:
                print(f"Warning: Could not create git tag: {e}")
        
        return data_version
    
    def load_version(self, version: Optional[str] = None) -> DataVersion:
        """
        Load data version (defaults to latest).
        
        Args:
            version: Specific version to load, or None for latest
        """
        if version:
            # Load specific version from git tag or version history
            return self._load_specific_version(version)
        else:
            # Load current version from .dataversion file
            if not self.version_file.exists():
                raise ValueError("No data version file found")
            
            with open(self.version_file) as f:
                data = json.load(f)
                return DataVersion(**data)
    
    def list_versions(self) -> List[DataVersion]:
        """List all available data versions"""
        versions = []
        
        # Get versions from git tags
        try:
            tags = subprocess.check_output(
                ["git", "tag", "-l", "data-v*"],
                cwd=self.data_root.parent
            ).decode().strip().split("\n")
            
            for tag in tags:
                if tag:
                    version = tag.replace("data-v", "")
                    try:
                        versions.append(self._load_specific_version(version))
                    except:
                        pass
        except:
            pass
        
        # Sort by version (semantic versioning)
        versions.sort(key=lambda v: self._version_key(v.version), reverse=True)
        
        return versions
    
    def verify_version(self, version: str, examples_path: Path) -> bool:
        """
        Verify that examples_path matches the checksum for version.
        Used to ensure data integrity.
        """
        data_version = self.load_version(version)
        current_checksum = self._compute_checksum(examples_path)
        
        return current_checksum == data_version.checksum
    
    def _compute_checksum(self, examples_path: Path) -> str:
        """Compute SHA256 checksum of all example files"""
        hasher = hashlib.sha256()
        
        for example_file in sorted(examples_path.rglob("*.md")):
            with open(example_file, "rb") as f:
                hasher.update(f.read())
        
        return hasher.hexdigest()
    
    def _count_examples(self, examples_path: Path) -> int:
        """Count number of example files"""
        return len(list(examples_path.rglob("*.md")))
    
    def _save_version(self, data_version: DataVersion):
        """Save version metadata to .dataversion file"""
        with open(self.version_file, "w") as f:
            json.dump(data_version.to_dict(), f, indent=2)
    
    def _load_specific_version(self, version: str) -> DataVersion:
        """Load version from git tag"""
        # Checkout tag and read .dataversion
        # This is a simplified version - in practice, you'd want
        # to cache version metadata or store in a separate index
        raise NotImplementedError("Loading specific versions requires git checkout")
    
    def _version_key(self, version: str) -> tuple:
        """Convert semantic version to sortable tuple"""
        parts = version.split(".")
        return tuple(int(p) for p in parts)
```

**File**: `pramana/data/dataset_registry.py`

```python
from pathlib import Path
from typing import Dict, Any, Optional
from .versioning import DataVersionManager, DataVersion

class DatasetRegistry:
    """
    Registry for tracking datasets and their versions.
    Provides high-level interface for data versioning.
    """
    
    def __init__(self, data_root: Path):
        self.data_root = Path(data_root)
        self.version_manager = DataVersionManager(self.data_root)
        self.registry_file = self.data_root / ".dataset_registry.json"
    
    def register_dataset(
        self,
        dataset_name: str,
        stage: int,
        version: str,
        examples_path: Path,
        quality_scores: Dict[str, float],
        changes: List[str]
    ) -> DataVersion:
        """
        Register a new dataset version.
        
        Args:
            dataset_name: Name of dataset (e.g., "stage_zero", "stage_one")
            stage: Training stage (0-3)
            version: Semantic version
            examples_path: Path to examples
            quality_scores: Quality metrics
            changes: Changelog
        """
        # Create version
        data_version = self.version_manager.create_version(
            version=version,
            examples_path=examples_path,
            quality_scores=quality_scores,
            changes=changes
        )
        
        # Register in registry
        self._add_to_registry(dataset_name, stage, version, data_version)
        
        return data_version
    
    def get_dataset_version(
        self,
        dataset_name: str,
        version: Optional[str] = None
    ) -> tuple[Path, DataVersion]:
        """
        Get dataset path and version metadata.
        
        Returns:
            (examples_path, data_version)
        """
        registry = self._load_registry()
        
        if dataset_name not in registry:
            raise ValueError(f"Dataset {dataset_name} not found")
        
        dataset_info = registry[dataset_name]
        
        # Get version (default to latest)
        if version:
            if version not in dataset_info["versions"]:
                raise ValueError(f"Version {version} not found for {dataset_name}")
            version_info = dataset_info["versions"][version]
        else:
            # Get latest version
            versions = sorted(
                dataset_info["versions"].keys(),
                key=lambda v: self._version_key(v),
                reverse=True
            )
            if not versions:
                raise ValueError(f"No versions found for {dataset_name}")
            version_info = dataset_info["versions"][versions[0]]
        
        examples_path = Path(version_info["examples_path"])
        data_version = DataVersion(**version_info["metadata"])
        
        return examples_path, data_version
    
    def _add_to_registry(
        self,
        dataset_name: str,
        stage: int,
        version: str,
        data_version: DataVersion
    ):
        """Add dataset version to registry"""
        registry = self._load_registry()
        
        if dataset_name not in registry:
            registry[dataset_name] = {
                "stage": stage,
                "versions": {}
            }
        
        registry[dataset_name]["versions"][version] = {
            "examples_path": str(self.data_root / dataset_name / f"v{version}"),
            "metadata": data_version.to_dict()
        }
        
        self._save_registry(registry)
    
    def _load_registry(self) -> Dict[str, Any]:
        """Load registry from file"""
        if not self.registry_file.exists():
            return {}
        
        with open(self.registry_file) as f:
            return json.load(f)
    
    def _save_registry(self, registry: Dict[str, Any]):
        """Save registry to file"""
        with open(self.registry_file, "w") as f:
            json.dump(registry, f, indent=2)
    
    def _version_key(self, version: str) -> tuple:
        """Convert semantic version to sortable tuple"""
        parts = version.split(".")
        return tuple(int(p) for p in parts)
```

---

## Implementation Map

### Files to Create

1. **Training Infrastructure**
   - `pramana/training/__init__.py`
   - `pramana/training/base_trainer.py` (BaseTrainer abstract class)
   - `pramana/training/sft_trainer.py` (SFTTrainerWrapper for Stages 0-2)
   - `pramana/training/grpo_trainer.py` (GRPOTrainerWrapper for Stage 3)
   - `pramana/training/checkpoint_manager.py` (CheckpointRepository)

2. **Reward Functions**
   - `pramana/rewards/__init__.py`
   - `pramana/rewards/composite_reward.py` (CompositeRewardFunction)
   - `pramana/rewards/nyaya_rewards.py` (Nyaya-specific rewards)

3. **Evaluation Pipeline**
   - `pramana/evaluation/__init__.py`
   - `pramana/evaluation/evaluation_pipeline.py` (EvaluationPipeline, handlers)
   - `pramana/evaluation/tier1_automated.py` (Tier1AutomatedHandler implementation)
   - `pramana/evaluation/tier2_llm_judge.py` (Tier2LLMJudgeHandler implementation)
   - `pramana/evaluation/tier3_manual.py` (Tier3ManualHandler implementation)

4. **Data Versioning**
   - `pramana/data/__init__.py`
   - `pramana/data/versioning.py` (DataVersionManager)
   - `pramana/data/dataset_registry.py` (DatasetRegistry)

5. **Validation Utilities** (referenced by rewards)
   - `pramana/validation/__init__.py`
   - `pramana/validation/structure_validator.py` (NyayaStructureValidator)
   - `pramana/validation/z3_verifier.py` (Z3Verifier)
   - `pramana/validation/hetvabhasa_detector.py` (HetvabhasaDetector)
   - `pramana/validation/pramana_scorer.py` (PramanaScorer)

### Files to Modify

1. **Configuration Files**
   - `configs/stage_zero_config.yaml` - Add trainer_class, reward_config
   - `configs/stage_one_config.yaml` - Add trainer_class, reward_config
   - `configs/stage_two_config.yaml` - Add trainer_class, reward_config
   - `configs/grpo_stage_three_config.yaml` - Add reward_function config

2. **Training Scripts** (if they exist)
   - Update to use new trainer abstractions
   - Integrate checkpoint manager
   - Add data version tracking

---

## Data Flow

### Training Flow (Stages 0-2: SFT)

```
1. Load Config (YAML)
   ↓
2. Initialize DataVersionManager → Load data version
   ↓
3. Initialize SFTTrainerWrapper with config + dataset
   ↓
4. BaseTrainer.train() template method:
   a. _setup_logging() → W&B init
   b. _load_model() → Unsloth FastLanguageModel
   c. _prepare_data() → Dataset loading
   d. _build_training_args() → TrainingArguments
   e. _build_trainer() → SFTTrainer (stage-specific)
   f. _run_training() → trainer.train()
   g. _save_final_checkpoint() → CheckpointRepository.save()
   h. _log_final_metrics() → W&B log
   ↓
5. CheckpointRepository saves:
   - Model weights
   - Tokenizer
   - Metadata (git commit, hyperparams, metrics)
   ↓
6. Return training results
```

### Training Flow (Stage 3: GRPO)

```
1. Load Config (YAML)
   ↓
2. Initialize CompositeRewardFunction with 5 components
   ↓
3. Initialize GRPOTrainerWrapper with config + reward function
   ↓
4. BaseTrainer.train() template method:
   a. _setup_logging()
   b. _load_model()
   c. _prepare_data() → Format prompts for GRPO
   d. _build_training_args() → GRPO-specific args
   e. _build_trainer() → GRPOTrainer with reward_function
   f. _run_training() → trainer.train() (GRPO iterations)
   g. _save_final_checkpoint()
   h. _log_final_metrics()
   ↓
5. CheckpointRepository saves with reward metrics
```

### Evaluation Flow

```
1. Model generates solution for problem
   ↓
2. EvaluationPipeline.evaluate(example, model_output)
   ↓
3. Tier1AutomatedHandler._evaluate():
   - NyayaStructureValidator.validate()
   - Format score + completeness score
   - If score < 0.7 → return failed, don't continue
   ↓
4. If Tier1 passed → Tier2LLMJudgeHandler._evaluate():
   - LLM judge evaluates quality
   - Component scores (reasoning, Nyaya adherence)
   - If score < 0.75 → return failed, don't continue
   ↓
5. If Tier2 passed → Tier3ManualHandler._evaluate():
   - Queue for manual review
   - Returns placeholder (review done offline)
   ↓
6. Aggregate results across tiers
   ↓
7. Log to evaluation database/metrics
```

### Reward Computation Flow (GRPO)

```
1. Model generates solution
   ↓
2. CompositeRewardFunction.compute(generated_solution, problem, ground_truth)
   ↓
3. For each RewardComponent:
   a. component.compute_fn() → raw score
   b. Apply weight → weighted_score
   c. Store in component_scores dict
   ↓
4. Sum weighted scores → total_reward
   ↓
5. Return (total_reward, component_scores)
   ↓
6. GRPOTrainer uses total_reward for policy update
   ↓
7. Log component_scores for debugging/analysis
```

### Checkpoint Management Flow

```
1. Training completes → CheckpointRepository.save_checkpoint()
   ↓
2. Create checkpoint directory (checkpoint_id)
   ↓
3. Copy model + tokenizer files
   ↓
4. Enrich metadata:
   - Get git commit (subprocess)
   - Get git branch (subprocess)
   - Get data version (from .dataversion)
   - Include hyperparameters
   - Include metrics
   ↓
5. Save metadata.json
   ↓
6. Update checkpoint index
   ↓
7. (Optional) Upload to W&B artifact store
```

### Data Versioning Flow

```
1. Create/update examples in data/seed_examples/stage_X/
   ↓
2. DataVersionManager.create_version():
   - Compute checksum of all .md files
   - Get git commit
   - Count examples
   - Create DataVersion object
   ↓
3. Save .dataversion file
   ↓
4. Create git tag (data-v1.0)
   ↓
5. DatasetRegistry.register_dataset():
   - Register in .dataset_registry.json
   - Link to version metadata
   ↓
6. Training scripts reference version:
   - dataset_registry.get_dataset_version("stage_zero", "1.0")
   - Returns (examples_path, data_version)
   ↓
7. Checkpoint metadata includes data_version
```

---

## Build Sequence

### Phase 1: Foundation (Week 1)

- [ ] **1.1** Create package structure (`pramana/training/`, `pramana/rewards/`, etc.)
- [ ] **1.2** Implement `BaseTrainer` abstract class
- [ ] **1.3** Implement `CheckpointRepository` with metadata enrichment
- [ ] **1.4** Implement `DataVersionManager` with Git integration
- [ ] **1.5** Create unit tests for checkpoint and versioning

### Phase 2: SFT Training (Week 2)

- [ ] **2.1** Implement `SFTTrainerWrapper` for Stages 0-2
- [ ] **2.2** Create validation utilities (`NyayaStructureValidator`, etc.)
- [ ] **2.3** Integrate with existing configs (YAML loading)
- [ ] **2.4** Test Stage 0 training end-to-end
- [ ] **2.5** Verify checkpoint saving/loading works

### Phase 3: Reward Functions (Week 3)

- [ ] **3.1** Implement `CompositeRewardFunction` base
- [ ] **3.2** Implement individual reward components:
  - [ ] Structure completeness
  - [ ] Z3 consistency
  - [ ] Hetvabhasa detection
  - [ ] Pramana appropriateness
  - [ ] Answer correctness
- [ ] **3.3** Create `create_nyaya_reward_function()` factory
- [ ] **3.4** Unit tests for reward components
- [ ] **3.5** Integration test with sample problems

### Phase 4: GRPO Training (Week 4)

- [ ] **4.1** Implement `GRPOTrainerWrapper` for Stage 3
- [ ] **4.2** Integrate `CompositeRewardFunction` with GRPO
- [ ] **4.3** Test GRPO training loop (small scale)
- [ ] **4.4** Verify reward logging and component tracking

### Phase 5: Evaluation Pipeline (Week 5)

- [ ] **5.1** Implement `EvaluationHandler` base class
- [ ] **5.2** Implement `Tier1AutomatedHandler`
- [ ] **5.3** Implement `Tier2LLMJudgeHandler` (LLM integration)
- [ ] **5.4** Implement `Tier3ManualHandler` (review queue)
- [ ] **5.5** Implement `EvaluationPipeline` orchestrator
- [ ] **5.6** Test evaluation chain (Tier1 → Tier2 → Tier3)
- [ ] **5.7** Verify early termination works correctly

### Phase 6: Integration & Testing (Week 6)

- [ ] **6.1** End-to-end test: Stage 0 training → checkpoint → evaluation
- [ ] **6.2** End-to-end test: Stage 3 GRPO → reward computation → checkpoint
- [ ] **6.3** Test checkpoint comparison and rollback
- [ ] **6.4** Test data versioning workflow (create → register → use)
- [ ] **6.5** Performance testing (checkpoint save/load speed)
- [ ] **6.6** Documentation and examples

---

## Critical Details

### Error Handling

- **Reward Components**: Individual component failures don't crash entire reward computation (catch exceptions, return 0.0)
- **Checkpoint Saving**: If metadata enrichment fails (git unavailable), save checkpoint anyway with partial metadata
- **Evaluation Pipeline**: Tier failures are logged but don't stop pipeline (graceful degradation)

### State Management

- **Checkpoint Metadata**: Immutable once saved (append-only log for changes)
- **Data Versions**: Immutable snapshots (new versions don't modify old ones)
- **Evaluation Results**: Stored in evaluation database with timestamps

### Performance Considerations

- **Checkpoint Storage**: Use symlinks for model files if disk space is limited (copy_files=False)
- **Reward Computation**: Cache expensive operations (Z3 verification results)
- **Evaluation Pipeline**: Parallelize Tier1 evaluation across batch
- **Data Versioning**: Checksum computation can be slow for large datasets (consider incremental hashing)

### Testing Strategy

- **Unit Tests**: Each component (trainer, reward, evaluation handler) tested independently
- **Integration Tests**: Full training loop with mock data
- **End-to-End Tests**: Real training run (small scale) with checkpoint save/load
- **Property Tests**: Reward function properties (monotonicity, normalization)

### Security Considerations

- **Git Integration**: Sanitize git command outputs (prevent code injection)
- **Checkpoint Metadata**: Don't log sensitive data (API keys, etc.)
- **Data Versioning**: Verify checksums before using data (prevent tampering)

---

## Example Usage

### Stage 0 Training

```python
from pramana.training import SFTTrainerWrapper, TrainerConfig, TrainingStage
from pramana.data import DatasetRegistry
from pathlib import Path

# Load dataset version
registry = DatasetRegistry(Path("data"))
examples_path, data_version = registry.get_dataset_version("stage_zero", "1.0")

# Load config
config = TrainerConfig(
    stage=TrainingStage.POC,
    experiment_name="pramana-stage-zero-v1",
    output_dir=Path("models/checkpoints/stage_zero"),
    base_model="unsloth/Meta-Llama-3.1-8B-bnb-4bit",
    wandb_project="pramana-stage-zero",
    lora_config={"r": 64, "lora_alpha": 16, ...}
)

# Create trainer
trainer = SFTTrainerWrapper(config, train_dataset, val_dataset)

# Train
results = trainer.train()

# Checkpoint automatically saved with metadata
```

### Stage 3 GRPO Training

```python
from pramana.training import GRPOTrainerWrapper, TrainerConfig, TrainingStage
from pramana.rewards import create_nyaya_reward_function

# Create reward function
reward_function = create_nyaya_reward_function()

# Create GRPO trainer
config = TrainerConfig(
    stage=TrainingStage.RL,
    experiment_name="pramana-grpo-v1",
    ...
)

trainer = GRPOTrainerWrapper(
    config=config,
    train_dataset=train_dataset,
    reward_function=reward_function,
    num_generations=4,
    beta=0.01
)

results = trainer.train()
```

### Evaluation

```python
from pramana.evaluation import EvaluationPipeline, Tier1AutomatedHandler, Tier2LLMJudgeHandler
from pramana.validation import NyayaStructureValidator

# Build evaluation pipeline
validator = NyayaStructureValidator()
tier1 = Tier1AutomatedHandler(validator)
tier2 = Tier2LLMJudgeHandler(llm_judge)
pipeline = EvaluationPipeline(tier1, tier2)

# Evaluate
results = pipeline.evaluate(example, model_output, max_tier=EvaluationTier.TIER2_LLM_JUDGE)

# Check results
tier1_result = results[0]
if tier1_result.passed:
    print(f"Tier 1 passed: {tier1_result.score}")
    tier2_result = results[1]
    print(f"Tier 2 score: {tier2_result.score}")
```

---

This architecture provides a solid foundation for implementing the Pramana training pipeline with clear separation of concerns, extensibility, and reproducibility.
