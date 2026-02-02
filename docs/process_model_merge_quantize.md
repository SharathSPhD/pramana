# Model Merge, Quantization, and Publishing

This runbook documents how to create a full merged model, convert to GGUF for
Ollama, and publish to Hugging Face.

## A) Merge LoRA adapter into base model (full weights)

**Inputs**:
- Base model ID (e.g., `unsloth/llama-3.2-3b-instruct`,
  `unsloth/DeepSeek-R1-Distill-Llama-8B`)
- LoRA adapter directory (e.g., `models/stage_0_corrected` or a HF adapter repo)

**Output**:
- Full merged model folder containing:
  - `model.safetensors` (or sharded safetensors)
  - `config.json`
  - `tokenizer.json` / `tokenizer_config.json`
  - `generation_config.json`

**Example (Python)**:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_id = "unsloth/llama-3.2-3b-instruct"
adapter_path = "models/stage_0_corrected"
out_dir = "hf_upload_full/nyaya-llama-3b-stage0-merged"

tokenizer = AutoTokenizer.from_pretrained(base_id, use_fast=True)
base = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype="auto")
merged = PeftModel.from_pretrained(base, adapter_path)
merged = merged.merge_and_unload()

merged.save_pretrained(out_dir, safe_serialization=True)
tokenizer.save_pretrained(out_dir)
```

**Notes**:
- This produces a standalone model; no adapter needed at inference time.
- Stage 1 full model artifacts are stored in:
  `/workspace/pramana/hf_upload_full/nyaya-deepseek-8b-stage1-merged`
- Stage 0 full model artifacts are stored in:
  `/workspace/pramana/hf_upload_full/nyaya-llama-3b-stage0-merged`

## B) Convert to GGUF (required by Ollama)

Ollama consumes `.gguf` files. Convert the full merged model to GGUF via
`llama.cpp` and then quantize.

### 1) Clone and build llama.cpp (in `pramana-unsloth`)
```bash
docker exec -it pramana-unsloth /bin/bash
cd /workspace
git clone https://github.com/ggerganov/llama.cpp
cd /workspace/llama.cpp
cmake -S . -B build
cmake --build build --config Release -j$(nproc)
```

### 2) Convert full model to GGUF (F16)
```bash
python /workspace/llama.cpp/convert_hf_to_gguf.py \
  /workspace/pramana/hf_upload_full/nyaya-deepseek-8b-stage1-merged \
  --outfile /workspace/pramana/hf_upload_full/nyaya-deepseek-8b-stage1-merged/nyaya-deepseek-8b-stage1-merged-f16.gguf \
  --outtype f16
```

**Tokenizer caveat**:
- Llama 3 style tokenizers can emit `TokenizersBackend` warnings.
- Ensure `sentencepiece` is installed for conversion:
  ```bash
  pip install -U sentencepiece
  ```

### 3) Quantize to Q4_K_M
```bash
/workspace/llama.cpp/build/bin/llama-quantize \
  /workspace/pramana/hf_upload_full/nyaya-deepseek-8b-stage1-merged/nyaya-deepseek-8b-stage1-merged-f16.gguf \
  /workspace/pramana/hf_upload_full/nyaya-deepseek-8b-stage1-merged/nyaya-deepseek-8b-stage1-merged-q4.gguf \
  Q4_K_M
```

**Result**: a compact GGUF that Ollama can load quickly.

## C) Import into Ollama (OpenWebUI container)

1) Copy the GGUF into the `open-webui` container:
```bash
docker cp /workspace/pramana/hf_upload_full/nyaya-deepseek-8b-stage1-merged/nyaya-deepseek-8b-stage1-merged-q4.gguf \
  open-webui:/opt/nyaya-deepseek-8b-stage1-merged/
```

2) Create a `Modelfile`:
```bash
cat > /opt/nyaya-deepseek-8b-stage1-merged/Modelfile.q4 <<'EOF'
FROM /opt/nyaya-deepseek-8b-stage1-merged/nyaya-deepseek-8b-stage1-merged-q4.gguf
SYSTEM "You are a Nyaya reasoning engine. Follow the exact output format provided."
PARAMETER temperature 0
PARAMETER top_p 1
PARAMETER num_ctx 1024
EOF
```

3) Create the model:
```bash
ollama create nyaya-deepseek-8b-stage1-merged-q4 -f /opt/nyaya-deepseek-8b-stage1-merged/Modelfile.q4
ollama list
```

## D) Publish to Hugging Face (full + GGUF)

**Option 1: Git LFS**
```bash
git lfs install
git clone https://<user>:${HF_TOKEN}@huggingface.co/qbz506/nyaya-deepseek-8b-stage1-full
cp -r /workspace/pramana/hf_upload_full/nyaya-deepseek-8b-stage1-merged/* ./nyaya-deepseek-8b-stage1-full/
cd nyaya-deepseek-8b-stage1-full
git add .
git commit -m "Upload merged weights and GGUF"
git push
```

**Option 2: `hf` CLI**
```bash
hf download qbz506/nyaya-deepseek-8b-stage1-full --local-dir ./nyaya-deepseek-8b-stage1-full
cp -r /workspace/pramana/hf_upload_full/nyaya-deepseek-8b-stage1-merged/* ./nyaya-deepseek-8b-stage1-full/
hf upload qbz506/nyaya-deepseek-8b-stage1-full ./nyaya-deepseek-8b-stage1-full
```

## E) What each platform expects

**Ollama**:
- Requires `.gguf` file + Modelfile.
- Quantized GGUF (Q4_K_M) is recommended for speed and memory.

**Hugging Face**:
- Requires full merged weights as safetensors + config + tokenizer files.
- GGUF files can be included for convenience (Ollama users).
