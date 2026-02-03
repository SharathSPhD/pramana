# Pramana Paper: LaTeX Compilation Guide

This directory contains the LaTeX source files for the Pramana research paper submitted to NeurIPS 2024.

## File Structure

```
docs/paper/
├── pramana_paper.tex          # Main LaTeX file
├── neurips_2024.sty            # NeurIPS 2024 style file
├── references.bib              # Bibliography database
├── sections/                   # Section files
│   ├── 01_abstract.tex
│   ├── 02_introduction.tex
│   ├── 03_related_work.tex
│   ├── 04_nyaya_framework.tex
│   ├── 05_methodology.tex
│   ├── 06_implementation.tex
│   ├── 07_results.tex
│   ├── 08_discussion.tex
│   ├── 09_open_source.tex
│   ├── 10_future_work.tex
│   ├── 11_conclusion.tex
│   └── appendices.tex
└── figures/                    # Figure PDFs
    ├── architecture.pdf
    ├── nyaya_flow.pdf
    ├── stage0_loss_curves.pdf
    ├── stage1_loss_curves.pdf
    ├── stage0_ablation.pdf
    ├── stage1_ablation.pdf
    ├── cross_stage_metrics.pdf
    └── parse_errors.pdf
```

## Compilation Instructions

### Prerequisites

The paper requires a LaTeX distribution with the following packages:
- `texlive-latex-base`
- `texlive-latex-extra`
- `texlive-fonts-recommended`
- `texlive-bibtex-extra`
- `texlive-science` (for algorithm package)

### Compilation Sequence

To compile the paper, run the following commands in order:

```bash
cd docs/paper

# First pass: Generate auxiliary files
pdflatex pramana_paper.tex

# Process bibliography
bibtex pramana_paper

# Second pass: Resolve citations
pdflatex pramana_paper.tex

# Third pass: Finalize all references
pdflatex pramana_paper.tex
```

### Docker Compilation

If compiling inside the `pramana-unsloth` Docker container:

```bash
docker exec pramana-unsloth bash -c "cd /workspace/pramana/docs/paper && pdflatex -interaction=nonstopmode pramana_paper.tex && bibtex pramana_paper && pdflatex -interaction=nonstopmode pramana_paper.tex && pdflatex -interaction=nonstopmode pramana_paper.tex"
```

## Required LaTeX Packages

The paper uses the following LaTeX packages (all included in standard TeX Live distributions):

**Core Packages:**
- `article` (document class)
- `graphicx` (figure inclusion)
- `amsmath`, `amssymb` (mathematics)
- `booktabs` (professional tables)
- `hyperref` (hyperlinks)
- `natbib` (bibliography)

**Additional Packages:**
- `algorithm`, `algorithmic` (algorithm pseudocode)
- `xcolor` (color support)
- `multirow` (table row spanning)
- `array` (advanced table formatting)
- `url` (URL formatting)
- `geometry` (page layout)
- `microtype` (microtypography)
- `caption` (caption formatting)
- `titlesec` (section formatting)

## Figure Regeneration

Figures are stored as PDF files in the `figures/` directory. To regenerate figures:

1. **Loss curves** (`stage0_loss_curves.pdf`, `stage1_loss_curves.pdf`): Generated from training logs using matplotlib/seaborn
2. **Architecture diagram** (`architecture.pdf`): Generated from Mermaid source (`architecture.mmd`) using `mmdc` or online Mermaid renderer
3. **Nyaya flow** (`nyaya_flow.pdf`): Generated from Mermaid source (`nyaya_flow.mmd`)
4. **Ablation plots** (`stage0_ablation.pdf`, `stage1_ablation.pdf`): Generated from evaluation results
5. **Cross-stage metrics** (`cross_stage_metrics.pdf`): Generated from comparison data
6. **Parse errors** (`parse_errors.pdf`): Generated from parse failure analysis

See `scripts/generate_paper_figures.py` for automated figure generation scripts.

## Known Issues and Warnings

### Minor Warnings (Non-Critical)

- **Overfull/Underfull hboxes**: Some paragraphs may have minor spacing issues. These are cosmetic and do not affect content.
- **Citation warnings**: During first compilation pass, citations will show as undefined. This is resolved after running `bibtex` and subsequent `pdflatex` passes.

### Style File Notes

- The `neurips_2024.sty` file is a custom style file based on NeurIPS formatting requirements
- The `\And` command for multiple authors is provided by the standard `article` class

## Output

After successful compilation:
- **PDF**: `pramana_paper.pdf` (approximately 67 pages, ~1.6 MB)
- **Auxiliary files**: `.aux`, `.bbl`, `.blg`, `.log` files are generated automatically

## Bibliography

The bibliography (`references.bib`) contains references organized by category:
1. Navya-Nyaya primary and computational works
2. LLM reasoning and chain-of-thought
3. Hallucination and verification
4. Neuro-symbolic AI
5. Fine-tuning frameworks
6. SMT solvers and formal verification
7. Additional relevant works

## Content Statistics

- **Total pages**: 67
- **Figures**: 8 (all PDF format)
- **Tables**: 25+ (across all sections)
- **Citations**: 51 unique citations
- **Sections**: 11 main sections + appendices

## Troubleshooting

### Missing Packages

If you encounter "File not found" errors for packages:
```bash
# On Ubuntu/Debian
sudo apt-get install texlive-latex-extra texlive-science

# Inside Docker container
apt-get update && apt-get install -y texlive-latex-extra texlive-science
```

### Bibliography Issues

If citations remain undefined after compilation:
1. Ensure `bibtex pramana_paper` was run (not `bibtex references`)
2. Check that `.bbl` file was generated
3. Run `pdflatex` two more times to resolve all references

### Figure Not Found

If figures fail to load:
1. Verify PDF files exist in `figures/` directory
2. Check that paths in `\includegraphics` commands are correct (relative to main `.tex` file)
3. Ensure PDF files are not corrupted

## Version Control

- LaTeX source files should be committed to version control
- Generated PDF (`pramana_paper.pdf`) should be committed for easy access
- Auxiliary files (`.aux`, `.log`, `.bbl`, etc.) should be in `.gitignore`
