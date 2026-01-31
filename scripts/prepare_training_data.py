#!/usr/bin/env python3
"""Prepare training data from seed examples.

Converts markdown seed examples to JSONL format for fine-tuning.
Each example contains:
- instruction: Problem statement
- input: Empty string (for consistency with training format)
- output: Full Nyaya reasoning trace from Samshaya to Nirnaya
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any


def extract_problem(content: str) -> str:
    """Extract the problem statement from markdown content.
    
    Args:
        content: Markdown content with YAML frontmatter removed
        
    Returns:
        Problem statement text
    """
    # Extract everything from "# Problem" to the next "##" section
    pattern = r"^#\s+Problem\s*\n(.*?)(?=^##\s+|\Z)"
    match = re.search(pattern, content, re.MULTILINE | re.DOTALL)
    
    if not match:
        raise ValueError("Missing '# Problem' section")
    
    problem_text = match.group(1).strip()
    return problem_text


def extract_reasoning_trace(content: str) -> str:
    """Extract the full Nyaya reasoning trace from Samshaya to Nirnaya.
    
    Args:
        content: Markdown content with YAML frontmatter removed
        
    Returns:
        Full reasoning trace from ## Samshaya to ## Nirnaya (inclusive)
    """
    # Find the start of Samshaya section
    samshaya_start = re.search(r"^##\s+Samshaya", content, re.MULTILINE)
    if not samshaya_start:
        raise ValueError("Missing '## Samshaya' section")
    
    # Find the end of Nirnaya section (everything until end of file or next top-level section)
    # We want everything from Samshaya to the end of Nirnaya
    start_pos = samshaya_start.start()
    
    # Extract from Samshaya to end of file (Nirnaya should be the last section)
    reasoning_trace = content[start_pos:].strip()
    
    # Verify Nirnaya section exists
    if "## Nirnaya" not in reasoning_trace:
        raise ValueError("Missing '## Nirnaya' section")
    
    return reasoning_trace


def remove_frontmatter(content: str) -> str:
    """Remove YAML frontmatter from markdown content.
    
    Args:
        content: Markdown content with YAML frontmatter
        
    Returns:
        Markdown content without frontmatter
    """
    # Match YAML frontmatter (between --- markers)
    pattern = r"^---\s*\n(.*?)^---\s*\n(.*)$"
    match = re.match(pattern, content, re.DOTALL | re.MULTILINE)
    
    if match:
        return match.group(2)
    return content  # No frontmatter found, return as-is


def parse_markdown_file(file_path: Path) -> dict[str, Any]:
    """Parse a markdown seed example file into training format.
    
    Args:
        file_path: Path to markdown file
        
    Returns:
        Dictionary with 'instruction', 'input', and 'output' keys
    """
    content = file_path.read_text(encoding="utf-8")
    
    # Remove YAML frontmatter
    content_no_frontmatter = remove_frontmatter(content)
    
    # Extract problem statement
    problem = extract_problem(content_no_frontmatter)
    
    # Extract full reasoning trace
    reasoning_trace = extract_reasoning_trace(content_no_frontmatter)
    
    return {
        "instruction": problem,
        "input": "",
        "output": reasoning_trace,
    }


def main() -> None:
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Convert markdown seed examples to JSONL training format"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input directory containing markdown seed examples",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSONL file path",
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_file = Path(args.output)
    
    # Validate input directory
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not input_dir.is_dir():
        raise ValueError(f"Input path is not a directory: {input_dir}")
    
    # Find all markdown files
    markdown_files = sorted(input_dir.glob("*.md"))
    
    if not markdown_files:
        raise ValueError(f"No markdown files found in {input_dir}")
    
    print(f"Found {len(markdown_files)} markdown file(s)")
    
    # Create output directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Process each file and write to JSONL
    with output_file.open("w", encoding="utf-8") as f:
        for md_file in markdown_files:
            try:
                example = parse_markdown_file(md_file)
                json_line = json.dumps(example, ensure_ascii=False)
                f.write(json_line + "\n")
                print(f"✓ Processed: {md_file.name}")
            except Exception as e:
                print(f"✗ Error processing {md_file.name}: {e}")
                raise
    
    print(f"\n✓ Successfully created training data: {output_file}")
    print(f"  Total examples: {len(markdown_files)}")


if __name__ == "__main__":
    main()
