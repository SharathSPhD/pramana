"""HuggingFace Hub uploader for models and datasets."""

import os
from pathlib import Path

from huggingface_hub import HfApi, create_repo, upload_folder


class HuggingFaceUploader:
    """Upload models and datasets to HuggingFace Hub."""

    def __init__(self, token: str | None = None) -> None:
        """Initialize HuggingFace uploader.

        Args:
            token: HuggingFace token. If None, uses HF_TOKEN environment variable.

        Raises:
            ValueError: If no token is provided and HF_TOKEN is not set.
        """
        self.token = token or os.getenv("HF_TOKEN")
        if not self.token:
            raise ValueError(
                "HuggingFace token is required. "
                "Provide it as an argument or set HF_TOKEN environment variable."
            )
        self._api = HfApi(token=self.token)

    def upload_model(
        self,
        model_path: Path,
        repo_id: str,
        private: bool = True,
    ) -> str:
        """Upload model to HuggingFace Hub.

        Args:
            model_path: Path to the model directory to upload.
            repo_id: Repository ID in format "org/repo-name".
            private: Whether the repository should be private.

        Returns:
            URL of the uploaded repository.

        Raises:
            ValueError: If model_path does not exist.
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise ValueError(f"Model path {model_path} does not exist.")

        # Generate README.md if not present
        self._ensure_readme(model_path, repo_id, repo_type="model")

        # Create repo if it doesn't exist
        if not self._api.repo_exists(repo_id=repo_id, repo_type="model", token=self.token):
            create_repo(
                repo_id=repo_id,
                repo_type="model",
                private=private,
                token=self.token,
                exist_ok=False,
            )

        # Upload folder
        upload_folder(
            folder_path=str(model_path),
            repo_id=repo_id,
            repo_type="model",
            token=self.token,
        )

        return f"https://huggingface.co/{repo_id}"

    def upload_dataset(
        self,
        data_path: Path,
        repo_id: str,
        private: bool = True,
    ) -> str:
        """Upload dataset to HuggingFace Hub.

        Args:
            data_path: Path to the dataset directory to upload.
            repo_id: Repository ID in format "org/repo-name".
            private: Whether the repository should be private.

        Returns:
            URL of the uploaded repository.

        Raises:
            ValueError: If data_path does not exist.
        """
        data_path = Path(data_path)
        if not data_path.exists():
            raise ValueError(f"Dataset path {data_path} does not exist.")

        # Generate README.md if not present
        self._ensure_readme(data_path, repo_id, repo_type="dataset")

        # Create repo if it doesn't exist
        if not self._api.repo_exists(repo_id=repo_id, repo_type="dataset", token=self.token):
            create_repo(
                repo_id=repo_id,
                repo_type="dataset",
                private=private,
                token=self.token,
                exist_ok=False,
            )

        # Upload folder
        upload_folder(
            folder_path=str(data_path),
            repo_id=repo_id,
            repo_type="dataset",
            token=self.token,
        )

        return f"https://huggingface.co/datasets/{repo_id}"

    def _ensure_readme(self, path: Path, repo_id: str, repo_type: str) -> None:
        """Ensure README.md exists in the given path.

        Args:
            path: Directory path where README.md should exist.
            repo_id: Repository ID for README content.
            repo_type: Type of repository ("model" or "dataset").
        """
        readme_path = path / "README.md"
        if not readme_path.exists():
            readme_content = self._generate_readme(repo_id, repo_type)
            readme_path.write_text(readme_content, encoding="utf-8")

    @staticmethod
    def _generate_readme(repo_id: str, repo_type: str) -> str:
        """Generate a basic README.md for the repository.

        Args:
            repo_id: Repository ID.
            repo_type: Type of repository ("model" or "dataset").

        Returns:
            README.md content as string.
        """
        title = repo_id.split("/")[-1].replace("-", " ").replace("_", " ").title()
        repo_type_label = repo_type.capitalize()

        return f"""---
license: mit
tags:
  - pramana
  - nyaya
  - reasoning
---

# {title}

This {repo_type_label.lower()} is part of the Pramana project - an epistemic reasoning engine for AI systems based on Navya-Nyaya logic.

## Repository

- **Repository ID**: `{repo_id}`
- **Type**: {repo_type_label}

## About Pramana

Pramana is a research project building an epistemic reasoning engine for AI systems based on 2,500-year-old Indian epistemological methodology (Navya-Nyaya logic). The goal is to create LLMs that apply structured 6-phase Nyaya methodology to solve logical problems systematically.

## Usage

See the [Pramana documentation](https://github.com/pramana/pramana) for usage instructions.

## Citation

If you use this {repo_type_label.lower()} in your research, please cite:

```bibtex
@software{{pramana,
  title={{Pramana: Epistemic Reasoning Engine}},
  author={{Pramana Team}},
  year={{2025}},
  url={{https://github.com/pramana/pramana}}
}}
```
"""
