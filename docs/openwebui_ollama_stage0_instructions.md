# Stage 0: OpenWebUI + Ollama Test Instructions

This document shows how to load the base model and the tuned Stage 0 LoRA adapter into Ollama/OpenWebUI, then test both with identical prompts.

Repository: `https://huggingface.co/qbz506/nyaya-llama-3b-stage0`

---

## What’s in the Hugging Face repo

The repo contains the LoRA adapter and supporting files (not a full base model):
- `adapter_model.safetensors`
- `adapter_config.json`
- `tokenizer.json`, `tokenizer_config.json`
- `chat_template.jinja`
- `docs/stage_0_review.md`
- `data/seed_examples/...`
- `results/stage_0_corrected_evaluation_v7.json`

Important: You must still pull the base model with Ollama. The tuned model is an adapter applied on top of the base.

---

## 1) Pull the base model in Ollama

Exact match base tag:

```bash
ollama pull llama3.2:3b
```

Verify:
```bash
ollama list
```

---

## 2) Download the tuned adapter from Hugging Face

### Option A — Using `hf` CLI

```bash
hf download qbz506/nyaya-llama-3b-stage0 --local-dir ./nyaya-llama-3b-stage0
```

### Option B — Using Git LFS

```bash
git lfs install
git clone https://huggingface.co/qbz506/nyaya-llama-3b-stage0
```

You should now have:
```
./nyaya-llama-3b-stage0/adapter_model.safetensors
./nyaya-llama-3b-stage0/adapter_config.json
```

---

## 3) Create the tuned Ollama model (LoRA adapter)

Create a `Modelfile` in the same directory as the adapter:

```bash
cd ./nyaya-llama-3b-stage0
cat > Modelfile <<'EOF'
FROM llama3.2:3b
ADAPTER ./
SYSTEM """
You are a Nyaya reasoning engine. Follow the exact output format provided.
"""
PARAMETER temperature 0
PARAMETER top_p 1
EOF
```

Create the model in Ollama:

```bash
ollama create nyaya-llama-3b-stage0 -f Modelfile
```

Verify:
```bash
ollama list
```

---

## 4) Load models in OpenWebUI

1. Open OpenWebUI
2. Go to Settings → Models
3. Click Refresh
4. You should see:
   - `llama3.2:3b`
   - `nyaya-llama-3b-stage0`

---

## 5) Test both models with identical prompts

### Recommended test prompt (user message)
Use the same user prompt for both models:

```text
Solve this using the 6-phase Nyaya methodology. Use the exact section headers:

## Samshaya (Doubt Analysis)
## Pramana (Sources of Knowledge)
## Pancha Avayava (5-Member Syllogism)
## Tarka (Counterfactual Reasoning)
## Hetvabhasa (Fallacy Check)
## Nirnaya (Ascertainment)

Problem:
Given: If P then Q. If Q then R. If R then S. P is true.
Question: What are the truth values of P, Q, R, and S?
```

### Same system prompt for both
For a fair comparison, keep the system prompt identical for both models. If you used the Modelfile SYSTEM block above, the tuned model already includes it. For the base model, set the same system prompt in OpenWebUI:

```text
You are a Nyaya reasoning engine. Follow the exact output format provided.
```

---

## Will the tuned model work without a system prompt?

Short answer: It may work, but format adherence is much stronger with a system prompt.

Why:
- The LoRA adapter was trained with explicit format instructions and templates.
- The system prompt is not embedded in the weights; it’s just a prompt.
- For maximum structure adherence, keep the system prompt (or embed it in the Modelfile).

---

## Optional: Strict template prompt
If you want maximal format adherence, use the full template from training in the user prompt:

```text
### Instructions:
You MUST follow the exact markdown structure below. Do NOT add "Phase" labels or alternative headings. Use the exact headers and field labels shown.

Required section order:
1) ## Samshaya (Doubt Analysis)
2) ## Pramana (Sources of Knowledge)
3) ## Pancha Avayava (5-Member Syllogism)
4) ## Tarka (Counterfactual Reasoning)
5) ## Hetvabhasa (Fallacy Check)
6) ## Nirnaya (Ascertainment)

### Template:
## Samshaya (Doubt Analysis)
**Doubt Type**:
**Justification**:

---

## Pramana (Sources of Knowledge)
### Pratyaksha (Direct Perception)
- 
### Anumana (Inference)
- 
### Upamana (Comparison)
- 
### Shabda (Testimony)
- 

---

## Pancha Avayava (5-Member Syllogism)
### Syllogism 1: 
**Pratijna (Thesis)**:
**Hetu (Reason)**:
**Udaharana (Universal + Example)**:
**Upanaya (Application)**:
**Nigamana (Conclusion)**:

---

## Tarka (Counterfactual Reasoning)
**Hypothesis**:
**Consequence**:
**Analysis**:
**Resolution**:

---

## Hetvabhasa (Fallacy Check)
Check for Savyabhichara: 
Check for Viruddha: 
Check for Asiddha: 
Check for Satpratipaksha: 
Check for Badhita: 

---

## Nirnaya (Ascertainment)
**Final Answer**:
**Justification**:
**Confidence**:
```

---

## Troubleshooting

- If `ollama create` fails, ensure the adapter path points to the folder containing `adapter_model.safetensors` and `adapter_config.json`.
- If OpenWebUI doesn’t show the model, refresh Models or restart OpenWebUI.
- If output ignores the Nyaya structure, apply the system prompt and the strict template.

---

## Quick Summary

- Base model: `ollama pull llama3.2:3b`
- Tuned model: use ADAPTER in a Modelfile and `ollama create`
- System prompt is not inside the weights; include it for best results
- Same prompt = best apples-to-apples comparison
# Stage 0: OpenWebUI + Ollama Test Instructions

This document shows how to load the **base model** and the **tuned Stage 0 LoRA adapter** into Ollama/OpenWebUI, then test both with identical prompts.

Repository: `https://huggingface.co/qbz506/nyaya-llama-3b-stage0`

---

## What’s in the Hugging Face repo

The repo contains the **LoRA adapter** and supporting files (not a full base model):
- `adapter_model.safetensors`
- `adapter_config.json`
- `tokenizer.json`, `tokenizer_config.json`
- `chat_template.jinja`
- `docs/stage_0_review.md`
- `data/seed_examples/...`
- `results/stage_0_corrected_evaluation_v7.json`

**Important**: You must still pull the **base model** with Ollama. The tuned model is an adapter applied on top of the base.

---

## 1) Pull the base model in Ollama

Exact match base tag:

```bash
ollama pull llama3.2:3b
```

Verify:
```bash
ollama list
```

---

## 2) Download the tuned adapter from Hugging Face

### Option A — Using `hf` CLI

```bash
hf download qbz506/nyaya-llama-3b-stage0 --local-dir ./nyaya-llama-3b-stage0
```

### Option B — Using Git LFS

```bash
git lfs install
git clone https://huggingface.co/qbz506/nyaya-llama-3b-stage0
```

You should now have:
```
./nyaya-llama-3b-stage0/adapter_model.safetensors
./nyaya-llama-3b-stage0/adapter_config.json
```

---

## 3) Create the tuned Ollama model (LoRA adapter)

Create a `Modelfile` in the same directory as the adapter:

```bash
cd ./nyaya-llama-3b-stage0
cat > Modelfile <<'EOF'
FROM llama3.2:3b
ADAPTER ./
SYSTEM """
You are a Nyaya reasoning engine. Follow the exact output format provided.
"""
PARAMETER temperature 0
PARAMETER top_p 1
