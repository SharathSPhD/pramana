"""Base trainer with Template Method pattern for training workflow."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from pramana.config.loader import StageConfig


@dataclass
class TrainingResult:
    """Training result containing metrics and checkpoint information."""

    final_loss: float
    best_checkpoint_path: Path
    metrics: dict[str, float]
    training_time_seconds: float


class BaseTrainer(ABC):
    """Template Method pattern for training workflow.

    This abstract base class defines the training workflow using the Template Method
    pattern. Subclasses must implement the abstract methods to provide stage-specific
    behavior.

    Example:
        ```python
        class SFTTrainer(BaseTrainer):
            def _setup(self, config: StageConfig) -> None:
                # Load model, setup tokenizer, etc.
                pass

            def _prepare_data(self) -> None:
                # Load and preprocess training data
                pass

            def _train(self) -> TrainingResult:
                # Run training loop
                return TrainingResult(...)
        ```
    """

    def train(self, config: StageConfig) -> TrainingResult:
        """Template method - defines training workflow.

        This method orchestrates the training process:
        1. Setup (model loading, configuration)
        2. Data preparation
        3. Training execution
        4. Cleanup (optional)

        Args:
            config: Stage configuration containing model, training, and data settings

        Returns:
            TrainingResult with final metrics and checkpoint path

        Raises:
            Any exceptions raised by _setup, _prepare_data, or _train are propagated
        """
        self._setup(config)
        self._prepare_data()
        result = self._train()
        self._cleanup()
        return result

    @abstractmethod
    def _setup(self, config: StageConfig) -> None:
        """Setup phase: load model, configure tokenizer, initialize training state.

        Args:
            config: Stage configuration

        Raises:
            Any setup-related exceptions should be raised here
        """
        ...

    @abstractmethod
    def _prepare_data(self) -> None:
        """Prepare data phase: load and preprocess training data.

        This method should load training data from the paths specified in the config,
        apply any necessary preprocessing, and prepare data loaders.

        Raises:
            Any data-related exceptions should be raised here
        """
        ...

    @abstractmethod
    def _train(self) -> TrainingResult:
        """Training phase: execute the training loop.

        This method contains the core training logic. It should:
        - Run the training loop
        - Track metrics
        - Save checkpoints
        - Return final results

        Returns:
            TrainingResult with final loss, checkpoint path, metrics, and training time

        Raises:
            Any training-related exceptions should be raised here
        """
        ...

    def _cleanup(self) -> None:  # noqa: B027
        """Optional cleanup phase: release resources, finalize logging.

        This hook is called after training completes (successfully or with error).
        Override this method to add cleanup logic such as:
        - Closing file handles
        - Releasing GPU memory
        - Finalizing experiment tracking

        By default, this method does nothing.
        """
        pass
