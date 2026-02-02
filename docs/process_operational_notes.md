# Operational Notes and Established Practices

This document captures additional practices that repeatedly came up during
Stage 0/Stage 1 operations and Space debugging.

## 1) Tokenizer compatibility (Llama 3)

Some Llama 3 checkpoints use `TokenizersBackend`. In Spaces, `AutoTokenizer`
may throw:

```
ValueError: Tokenizer class TokenizersBackend does not exist or is not currently imported.
```

Mitigation (in Space app):
- Try `AutoTokenizer` first
- On `TokenizersBackend` error, fall back to `LlamaTokenizerFast`

This is already implemented in `app.py`.

## 2) ZeroGPU constraints and mitigation

If the Space returns `"GPU task aborted"`:
- Split base and tuned generation into separate GPU tasks
- Reduce `max_new_tokens` (default 256 for Stage 1)
- Avoid caching large models in a single process

The current Space app uses separate GPU endpoints:
- `/generate_base`
- `/generate_tuned`

## 3) Ollama in OpenWebUI

Ollama expects GGUF. Process:
1. Convert and quantize in `pramana-unsloth`
2. Copy GGUF into `open-webui`
3. Create a Modelfile and run `ollama create`
4. Verify with `ollama list` and `ollama run`

Ollama model store:
- `/root/.ollama` in `open-webui`

## 4) Cross-container file movement

Use `docker cp` to move large artifacts:
```bash
docker cp pramana-unsloth:/workspace/pramana/hf_upload_full/.../model-q4.gguf /tmp/
docker cp /tmp/model-q4.gguf open-webui:/opt/nyaya-.../
```

## 5) HF token handling

HF_TOKEN is required for:
- Downloading models/datasets in the Space
- Git pushing to Hugging Face

Pattern used:
- `HF_TOKEN=$(docker exec pramana-dev printenv HF_TOKEN)` on host
- Use token in `git clone https://<user>:${HF_TOKEN}@huggingface.co/...`

## 6) HF Space development layout

Inside `open-webui`:
- `/opt/pramana-nyaya-demo/` (working files)
- `/opt/pramana-nyaya-demo-repo/` (git clone for commit/push)

Always push from the git clone directory.
