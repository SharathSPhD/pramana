#!/usr/bin/env bash
# =============================================================================
# setup_ollama_models.sh — One-time Ollama model setup for Pramana notebooks
#
# Pulls base models from the Ollama registry and creates tuned models from
# project GGUF files (downloaded from HuggingFace). Run this once before using
# the Pramana Explorer or Evaluation notebooks with the Ollama backend.
#
# Usage:
#   chmod +x setup_ollama_models.sh
#   ./setup_ollama_models.sh              # Set up all models
#   ./setup_ollama_models.sh --stage0     # Stage 0 models only
#   ./setup_ollama_models.sh --stage1     # Stage 1 models only
#
# Requirements:
#   - Ollama installed (https://ollama.com/install.sh)
#   - curl
#   - ~6 GB disk space for Stage 0, ~10 GB for Stage 1
# =============================================================================

set -euo pipefail

# ---------- Configuration ----------

# Stage 0: Llama 3.2 3B
STAGE0_BASE_TAG="llama3.2:3b-instruct-q4_K_M"
STAGE0_TUNED_TAG="nyaya-llama-3b-stage0"
STAGE0_GGUF_URL="https://huggingface.co/qbz506/nyaya-llama-3b-stage0-full/resolve/main/nyaya-llama-3b-stage0-merged-q4.gguf"

# Stage 1: DeepSeek-R1 Distill Llama 8B
STAGE1_BASE_TAG="deepseek-r1:8b-llama-distill-q4_K_M"
STAGE1_TUNED_TAG="nyaya-deepseek-8b-stage1"
# Stage 1 tuned GGUF not yet available — set to NONE to skip
STAGE1_GGUF_URL="NONE"

GGUF_DIR="/tmp/pramana_gguf"

# ---------- Helpers ----------

log()  { echo -e "\033[1;34m[INFO]\033[0m $*"; }
ok()   { echo -e "\033[1;32m[OK]\033[0m   $*"; }
warn() { echo -e "\033[1;33m[WARN]\033[0m $*"; }
err()  { echo -e "\033[1;31m[ERR]\033[0m  $*"; }

check_ollama() {
    if ! command -v ollama &>/dev/null; then
        err "Ollama not found. Install it first:"
        echo "  curl -fsSL https://ollama.com/install.sh | sh"
        exit 1
    fi
    # Ensure server is reachable
    if ! curl -sf http://localhost:11434/api/tags &>/dev/null; then
        log "Starting Ollama server in background..."
        OLLAMA_HOST=0.0.0.0 nohup ollama serve &>/dev/null &
        sleep 3
        if ! curl -sf http://localhost:11434/api/tags &>/dev/null; then
            err "Could not start Ollama server. Start it manually: ollama serve"
            exit 1
        fi
    fi
    ok "Ollama server is running"
}

model_exists() {
    # Returns 0 if the model tag already exists locally
    # Matches "modelname:tag" or "modelname " at start of line
    ollama list 2>/dev/null | grep -qE "^$1(:[^ ]+)?[[:space:]]" 2>/dev/null
}

pull_base_model() {
    local tag="$1"
    if model_exists "$tag"; then
        ok "Base model '$tag' already pulled"
    else
        log "Pulling base model: $tag (this may take a few minutes)..."
        ollama pull "$tag"
        ok "Base model '$tag' pulled"
    fi
}

download_gguf() {
    local name="$1"
    local url="$2"
    local dest="$GGUF_DIR/${name}.gguf"

    mkdir -p "$GGUF_DIR"

    if [[ -f "$dest" ]]; then
        ok "GGUF already downloaded: $dest" >&2
    else
        log "Downloading GGUF: $name ..." >&2
        curl -fSL --progress-bar -o "$dest" "$url" >&2
        ok "GGUF downloaded: $dest" >&2
    fi
    # Only the path goes to stdout (for capture by caller)
    echo "$dest"
}

create_tuned_model() {
    local tag="$1"
    local gguf_path="$2"

    if model_exists "$tag"; then
        ok "Tuned model '$tag' already exists"
        return
    fi

    local modelfile
    modelfile=$(mktemp /tmp/Modelfile.XXXXXX)
    cat > "$modelfile" <<MODELFILE
FROM ${gguf_path}
SYSTEM """
You are a Nyaya reasoning engine. Follow the exact output format provided.
Use the exact section headers:
## Samshaya (Doubt Analysis)
## Pramana (Sources of Knowledge)
## Pancha Avayava (5-Member Syllogism)
## Tarka (Counterfactual Reasoning)
## Hetvabhasa (Fallacy Check)
## Nirnaya (Ascertainment)
"""
PARAMETER temperature 0
PARAMETER num_ctx 4096
MODELFILE

    log "Creating tuned model: $tag ..."
    ollama create "$tag" -f "$modelfile"
    rm -f "$modelfile"
    ok "Tuned model '$tag' created"
}

# ---------- Stage setup functions ----------

setup_stage0() {
    log "===== Stage 0: Llama 3.2 3B ====="

    # Base model (pull from registry)
    pull_base_model "$STAGE0_BASE_TAG"

    # Tuned model (download GGUF + create)
    local gguf_path
    gguf_path=$(download_gguf "$STAGE0_TUNED_TAG" "$STAGE0_GGUF_URL")
    create_tuned_model "$STAGE0_TUNED_TAG" "$gguf_path"

    echo ""
}

setup_stage1() {
    log "===== Stage 1: DeepSeek-R1 Distill Llama 8B ====="

    # Base model (pull from registry)
    pull_base_model "$STAGE1_BASE_TAG"

    # Tuned model (download GGUF + create)
    if [[ -z "$STAGE1_GGUF_URL" || "$STAGE1_GGUF_URL" == "NONE" ]]; then
        warn "Stage 1 tuned model GGUF not yet available on HuggingFace."
        warn "The tuned model must be built from safetensors first."
        warn "Skipping tuned model setup for Stage 1."
    else
        local gguf_path
        gguf_path=$(download_gguf "$STAGE1_TUNED_TAG" "$STAGE1_GGUF_URL")
        create_tuned_model "$STAGE1_TUNED_TAG" "$gguf_path"
    fi

    echo ""
}

# ---------- Main ----------

main() {
    echo ""
    echo "=========================================="
    echo " Pramana — Ollama Model Setup"
    echo "=========================================="
    echo ""

    check_ollama

    local do_stage0=true
    local do_stage1=true

    for arg in "$@"; do
        case "$arg" in
            --stage0) do_stage1=false ;;
            --stage1) do_stage0=false ;;
            --help|-h)
                echo "Usage: $0 [--stage0] [--stage1]"
                echo "  --stage0  Set up Stage 0 models only"
                echo "  --stage1  Set up Stage 1 models only"
                echo "  (default: set up all models)"
                exit 0
                ;;
            *)
                err "Unknown argument: $arg"
                exit 1
                ;;
        esac
    done

    if $do_stage0; then setup_stage0; fi
    if $do_stage1; then setup_stage1; fi

    echo ""
    log "All done! Installed models:"
    ollama list
    echo ""
    ok "You can now use the Pramana notebooks with the Ollama backend."
}

main "$@"
