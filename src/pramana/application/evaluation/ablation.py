"""Utilities for shortcut detection ablation studies."""

from __future__ import annotations

import re


def build_nyaya_prompt(
    *, problem: str, format_instructions: str, format_template: str
) -> str:
    """Build a Nyaya-structured prompt for evaluation."""
    return f"""### Problem:
{problem}

### Instructions:
{format_instructions}

### Template:
{format_template}

### Nyaya Reasoning:
"""


def build_baseline_prompt(problem: str) -> str:
    """Build a baseline prompt without Nyaya structure."""
    return f"""Solve the problem step by step and provide a final answer.

Problem:
{problem}

Answer:
"""


def extract_answer_from_output(output: str) -> str:
    """Extract an answer from free-form model output."""
    match = re.search(
        r"(?:Final Answer|Answer)\s*:\s*(.+)", output, re.IGNORECASE
    )
    if match:
        return match.group(1).strip().splitlines()[0].strip()

    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines:
        return ""
    return lines[-1]


__all__ = ["build_baseline_prompt", "build_nyaya_prompt", "extract_answer_from_output"]
