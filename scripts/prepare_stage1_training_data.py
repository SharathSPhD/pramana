#!/usr/bin/env python3
"""Prepare Stage 1 training data from Stage 0 + Stage 1 seed examples."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


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
    reasoning_trace = content[samshaya_start.start() :].strip()
    if "## Nirnaya" not in reasoning_trace:
        raise ValueError("Missing '## Nirnaya' section")
    return reasoning_trace


def remove_frontmatter(content: str) -> str:
    pattern = r"^---\s*\n(.*?)^---\s*\n(.*)$"
    match = re.match(pattern, content, re.DOTALL | re.MULTILINE)
    if match:
        return match.group(2)
    return content


def parse_markdown_file(file_path: Path) -> dict[str, Any]:
    content = file_path.read_text(encoding="utf-8")
    content_no_frontmatter = remove_frontmatter(content)
    problem = extract_problem(content_no_frontmatter)
    reasoning_trace = extract_reasoning_trace(content_no_frontmatter)
    return {
        "instruction": problem,
        "input": "",
        "output": reasoning_trace,
    }


def collect_examples(input_dirs: list[Path]) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for directory in input_dirs:
        for md_file in sorted(directory.glob("*.md")):
            if md_file.name.lower() == "readme.md":
                continue
            examples.append(parse_markdown_file(md_file))
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare Stage 1 training data from seed examples."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/training/stage_1.jsonl",
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--stage-zero-dir",
        type=str,
        default="data/seed_examples/stage_zero",
        help="Stage 0 seed examples directory",
    )
    parser.add_argument(
        "--stage-one-dir",
        type=str,
        default="data/seed_examples/stage_one",
        help="Stage 1 seed examples directory",
    )
    args = parser.parse_args()

    input_dirs = [Path(args.stage_zero_dir), Path(args.stage_one_dir)]
    for directory in input_dirs:
        if not directory.exists():
            raise FileNotFoundError(f"Missing input directory: {directory}")

    examples = collect_examples(input_dirs)
    if not examples:
        raise ValueError("No examples collected.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(f"Wrote {len(examples)} examples to {output_path}")


if __name__ == "__main__":
    main()
