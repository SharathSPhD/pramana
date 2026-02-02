# Docker Runtime Map (Pramana)

This runbook documents the working Docker containers, what they do, and the
important paths/volumes used in this project.

## Current working containers

### 1) `pramana-unsloth`
- **Image**: `nvcr.io/nvidia/pytorch:25.11-py3`
- **Primary use**: Training, model merging, GGUF conversion, and any GPU-heavy
  Python workflows (Unsloth, transformers, peft, llama.cpp conversion).
- **Repo mount**: `/workspace/pramana` (bind-mounted from host repo root).
- **HF cache**: `/root/.cache/huggingface` (volume `pramana-hf-cache`).
- **Key folders inside**:
  - `/workspace/pramana/hf_upload_full/` (full merged model artifacts)
  - `/workspace/pramana/hf_upload/` (LoRA adapters and training artifacts)
  - `/workspace/pramana/scripts/` (training and evaluation scripts)
  - `/workspace/llama.cpp/` (llama.cpp clone for GGUF conversion/quantization)
- **Reason this is the "working" container**: It has GPU support, the repo is
  mounted, and it is the best place to run all training/merge/quantization steps.

### 2) `open-webui`
- **Image**: `ghcr.io/open-webui/open-webui:ollama`
- **Primary use**: OpenWebUI UI, Ollama model management, and Hugging Face Space
  app updates/pushes.
- **Ports**: `12000 -> 8080` (OpenWebUI)
- **Volumes**:
  - `/app/backend/data` (OpenWebUI data)
  - `/root/.ollama` (Ollama model store)
- **Key folders inside**:
  - `/opt/pramana-nyaya-demo/` (working copy of Space app files)
  - `/opt/pramana-nyaya-demo-repo/` (git clone of HF Space repo for commits/push)
  - `/opt/nyaya-llama-3b-stage0-merged/` (Stage 0 merged artifacts)
  - `/opt/nyaya-deepseek-8b-stage1-merged/` (Stage 1 merged artifacts)
- **Why this container is used for pushes**: It already has git and HF token
  access patterns established, and is the same environment used to load/verify
  Ollama models and HF Space changes.

### 3) `pramana-dev` (not the primary workflow container)
- **Image**: `pramana-pramana` (from `docker-compose.yml`)
- **Primary use**: Optional dev shell and Jupyter/TensorBoard ports.
- **Ports**: `8888`, `6006`
- **Note**: The main workflows use `pramana-unsloth` and `open-webui`.

## Quick operational commands

```bash
# List containers
docker ps -a

# Enter the working training/merge container
docker exec -it pramana-unsloth /bin/bash

# Enter OpenWebUI/Ollama container
docker exec -it open-webui /bin/bash

# Inspect mounts (useful for volumes and paths)
docker inspect pramana-unsloth --format '{{json .Mounts}}'
docker inspect open-webui --format '{{json .Mounts}}'
```

## Data location summary

- **Repo root (inside pramana-unsloth)**: `/workspace/pramana`
- **HF merged model artifacts**: `/workspace/pramana/hf_upload_full`
- **Ollama model store**: `/root/.ollama` (inside `open-webui`)
- **HF Space working copy**: `/opt/pramana-nyaya-demo` (inside `open-webui`)
- **HF Space git clone**: `/opt/pramana-nyaya-demo-repo` (inside `open-webui`)
