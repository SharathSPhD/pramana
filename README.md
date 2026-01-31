# Pramana

**Epistemic Reasoning Engine for AI Systems Based on Navya-Nyaya Logic**

Pramana is a research project that builds an epistemic reasoning engine for AI systems. It teaches LLMs to apply the 2,500-year-old Indian epistemological methodology of Navya-Nyaya logic to solve logical problems systematically, rather than relying on probabilistic pattern-matching.

## Overview

Unlike standard chain-of-thought reasoning, Pramana enforces a structured 6-phase Nyaya methodology:

1. **Samshaya** (Doubt Analysis) - Classify the type of uncertainty/ambiguity
2. **Pramana** (Evidence Sources) - Identify valid knowledge sources (Pratyaksha/perception, Anumana/inference, Upamana/comparison, Shabda/testimony)
3. **Pancha Avayava** (5-Member Syllogism) - Construct formal argument with Pratijna (thesis), Hetu (reason), Udaharana (universal example), Upanaya (application), Nigamana (conclusion)
4. **Tarka** (Counterfactual Testing) - Use reductio ad absurdum to verify conclusions
5. **Hetvabhasa** (Fallacy Detection) - Check for reasoning errors
6. **Nirnaya** (Ascertainment) - Reach definitive conclusion or explicitly state insufficient evidence

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/pramana.git
cd pramana

# Install with uv (recommended)
uv sync --dev

# Or install with pip
pip install -e ".[dev]"
```

## Development

```bash
# Run tests
uv run pytest

# Run linter
uv run ruff check

# Run type checker
uv run mypy src

# Format code
uv run ruff format
```

## Project Structure

```
pramana/
├── src/pramana/
│   ├── domain/          # Core business logic
│   ├── application/     # Use cases and orchestration
│   ├── infrastructure/  # External integrations
│   ├── config/          # Configuration management
│   └── cli/             # Command-line interface
├── tests/               # Test suite
├── configs/             # YAML configuration files
├── data/                # Training and evaluation data
└── docs/                # Documentation
```

## License

MIT License - see LICENSE for details.
