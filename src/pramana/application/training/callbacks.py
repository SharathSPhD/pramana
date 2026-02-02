"""Training callbacks for observability."""

from __future__ import annotations

import re
from dataclasses import dataclass

try:
    import torch
except ImportError:  # pragma: no cover - optional in some environments
    torch = None  # type: ignore[assignment]

try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None  # type: ignore[assignment]

try:
    from transformers import TrainerCallback
except ImportError:  # pragma: no cover - optional dependency
    TrainerCallback = object  # type: ignore[assignment]


@dataclass(frozen=True)
class NyayaSampleMetrics:
    """Simple structural metrics for a generated Nyaya sample."""

    format_adherence: float
    phase_count: int
    syllogism_count: int


class NyayaMetricsCallback(TrainerCallback):
    """Log Nyaya structural metrics during evaluation."""

    def __init__(
        self,
        *,
        tokenizer,
        prompt: str,
        max_new_tokens: int = 512,
        log_sample_text: bool = True,
    ) -> None:
        self._tokenizer = tokenizer
        self._prompt = prompt
        self._max_new_tokens = max_new_tokens
        self._log_sample_text = log_sample_text

    def on_evaluate(self, args, state, control, model=None, **kwargs):  # type: ignore[override]
        if model is None or torch is None:
            return

        model.eval()
        inputs = self._tokenizer(
            self._prompt,
            return_tensors="pt",
            truncation=True,
            max_length=getattr(args, "max_seq_length", None),
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                temperature=0.0,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        generated_tokens = outputs[0][inputs["input_ids"].shape[1] :]
        generated_part = self._tokenizer.decode(
            generated_tokens, skip_special_tokens=True
        ).strip()

        metrics = self._compute_metrics(generated_part)

        report_to = getattr(args, "report_to", None)
        wandb_enabled = False
        if isinstance(report_to, str):
            wandb_enabled = report_to == "wandb"
        elif isinstance(report_to, (list, tuple, set)):
            wandb_enabled = "wandb" in report_to

        if wandb is not None and wandb_enabled:
            payload = {
                "val/format_adherence": metrics.format_adherence,
                "val/phase_count": metrics.phase_count,
                "val/syllogism_count": metrics.syllogism_count,
            }
            if self._log_sample_text:
                payload["val/sample_text"] = wandb.Html(
                    f"<pre>{generated_part}</pre>"
                )
            wandb.log(payload, step=state.global_step)

        model.train()

    @staticmethod
    def _compute_metrics(sample: str) -> NyayaSampleMetrics:
        if not sample:
            return NyayaSampleMetrics(0.0, 0, 0)

        phases = [
            "Samshaya",
            "Pramana",
            "Pancha Avayava",
            "Tarka",
            "Hetvabhasa",
            "Nirnaya",
        ]
        phase_count = sum(
            1
            for phase in phases
            if re.search(rf"^##\s+{re.escape(phase)}", sample, re.MULTILINE)
        )
        format_adherence = phase_count / len(phases)
        syllogism_count = len(
            re.findall(r"^###\s+Syllogism", sample, re.MULTILINE)
        )

        return NyayaSampleMetrics(
            format_adherence=format_adherence,
            phase_count=phase_count,
            syllogism_count=syllogism_count,
        )
