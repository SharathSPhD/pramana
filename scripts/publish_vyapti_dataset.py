#!/usr/bin/env python3
"""Publish the Vyapti Probe Benchmark to Hugging Face Hub.

Usage:
    python scripts/publish_vyapti_dataset.py --repo qbz506/vyapti-probe-benchmark

Requires: huggingface_hub
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description="Publish vyapti benchmark to HF")
    parser.add_argument("--repo", default="qbz506/vyapti-probe-benchmark", help="HF repo ID")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be uploaded")
    args = parser.parse_args()

    data_dir = PROJECT_ROOT / "data" / "vyapti_probe"

    files_to_upload = [
        data_dir / "problems.json",
        data_dir / "solutions.json",
        data_dir / "README.md",
    ]

    # Also include Z3 encodings
    z3_dir = data_dir / "z3_encodings"
    if z3_dir.exists():
        for f in sorted(z3_dir.glob("*.py")):
            files_to_upload.append(f)

    # Include results if available
    results_dir = data_dir / "results"
    if results_dir.exists():
        for f in sorted(results_dir.glob("*.json")):
            files_to_upload.append(f)

    if args.dry_run:
        print(f"Would upload to {args.repo}:")
        for f in files_to_upload:
            rel = f.relative_to(data_dir)
            print(f"  {rel} ({f.stat().st_size / 1024:.1f} KB)")
        return

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("Install huggingface_hub: pip install huggingface_hub")
        sys.exit(1)

    api = HfApi()

    # Create repo if needed
    try:
        api.create_repo(args.repo, repo_type="dataset", exist_ok=True)
        print(f"Repository {args.repo} ready")
    except Exception as e:
        print(f"Error creating repo: {e}")
        sys.exit(1)

    # Upload files
    for f in files_to_upload:
        rel_path = str(f.relative_to(data_dir))
        print(f"Uploading {rel_path}...", end=" ")
        try:
            api.upload_file(
                path_or_fileobj=str(f),
                path_in_repo=rel_path,
                repo_id=args.repo,
                repo_type="dataset",
            )
            print("OK")
        except Exception as e:
            print(f"FAILED: {e}")

    print(f"\nDataset published: https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
