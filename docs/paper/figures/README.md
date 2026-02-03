# Architecture and Flow Diagrams

This directory contains two key diagrams for the Pramana paper.

## Diagram 1: System Architecture (`architecture.pdf`)

**Purpose**: Layered architecture diagram showing the complete system structure.

**Layers**:
- **Layer 1 (CLI)**: Entry points including `train`, `evaluate`, `validate`, and `deploy` commands
- **Layer 2 (Application)**: Core application logic including `MarkdownParser` (data loading), `EvaluationPipeline` (Chain of Responsibility pattern), and `TrainingOrchestrator` (workflow management)
- **Layer 3 (Domain)**: Domain logic including `NyayaStructureValidator` (6-phase validation), `SemanticMatcher` (content quality), and `FallacyDetector` (Hetvabhasa checks)
- **Layer 4 (Infrastructure)**: External integrations including `UnslothAdapter` (training framework), `Z3Verifier` (logical verification), and `HuggingFaceRepository` (model storage)

**Key Features**:
- Shows clear dependencies between layers
- Illustrates data flow from CLI through application to domain and infrastructure
- Color-coded layers for visual clarity
- Suitable for NeurIPS 2-column format

## Diagram 2: Nyaya 6-Phase Reasoning Flow (`nyaya_flow.pdf`)

**Purpose**: Complete flowchart of the Nyaya reasoning methodology with decision points and feedback loops.

**Phases**:
1. **Samshaya** (Doubt Analysis): Identify uncertainty type (ambiguity, contradiction, insufficient info)
2. **Pramana** (Evidence Sources): Select appropriate pramana type (Pratyaksha, Anumana, Upamana, Shabda)
3. **Pancha Avayava** (5-Member Syllogism): Construct formal argument with Pratijna → Hetu → Udaharana (with Vyapti) → Upanaya → Nigamana
4. **Tarka** (Counterfactual Testing): Reductio ad absurdum verification with feedback loop
5. **Hetvabhasa** (Fallacy Detection): Check for 5 fallacy types (Savyabhichara, Viruddha, Prakaranasama, Sadhyasama, Kalaatita) with correction feedback
6. **Nirnaya** (Ascertainment): Reach definitive conclusion or explicitly state insufficient evidence

**Key Features**:
- Shows decision points at each phase
- Illustrates feedback loops for error correction (Tarka and Hetvabhasa phases)
- Color-coded phase boxes, decision nodes, and feedback paths
- Demonstrates the critical path from problem input to final answer
- Suitable for NeurIPS 2-column format

## Generation

Diagrams are created as Mermaid source files (`.mmd`) and converted to PDF using the `convert_to_pdf.py` script, which uses the mermaid.ink API to generate images and reportlab to create PDFs.

## Usage in Paper

Both diagrams are designed to fit within NeurIPS 2-column format with appropriate margins and scaling. They use professional styling with minimal colors and clear visual hierarchy.
