# Hugging Face Space Operations

This runbook documents how to update, build, run, and test the HF Space:

`https://huggingface.co/spaces/qbz506/pramana-nyaya-demo`

## A) Local Space repo layout (OpenWebUI container)

Inside the `open-webui` container:
- **Working copy**: `/opt/pramana-nyaya-demo/`
- **Git clone**: `/opt/pramana-nyaya-demo-repo/`

Recommended workflow:
1. Edit or copy files into `/opt/pramana-nyaya-demo/`
2. Sync into `/opt/pramana-nyaya-demo-repo/`
3. Commit + push from the git clone

## B) Update and push workflow

```bash
docker exec -it open-webui /bin/bash

# In open-webui
cd /opt
rm -rf /opt/pramana-nyaya-demo-repo
git clone https://<user>:${HF_TOKEN}@huggingface.co/spaces/qbz506/pramana-nyaya-demo /opt/pramana-nyaya-demo-repo

# Copy updated files into repo
cp /opt/pramana-nyaya-demo/app.py /opt/pramana-nyaya-demo-repo/app.py
cp /opt/pramana-nyaya-demo/README.md /opt/pramana-nyaya-demo-repo/README.md
cp /opt/pramana-nyaya-demo/requirements.txt /opt/pramana-nyaya-demo-repo/requirements.txt

# Commit + push (local identity)
cd /opt/pramana-nyaya-demo-repo
git add app.py README.md requirements.txt
git -c user.name="pramana-bot" -c user.email="pramana@local" commit -m "Update Space app"
git push
```

## C) Build and run logs

Build logs:
```bash
curl -N \
  -H "Authorization: Bearer $HF_TOKEN" \
  "https://huggingface.co/api/spaces/qbz506/pramana-nyaya-demo/logs/build"
```

Run logs:
```bash
curl -N \
  -H "Authorization: Bearer $HF_TOKEN" \
  "https://huggingface.co/api/spaces/qbz506/pramana-nyaya-demo/logs/run"
```

Notes:
- Logs are streaming. Use `--max-time` if you need a bounded call.
- If logs reset, restart the Space to regenerate logs.

## D) Restart the Space (force new runtime)

```bash
curl -X POST \
  -H "Authorization: Bearer $HF_TOKEN" \
  "https://huggingface.co/api/spaces/qbz506/pramana-nyaya-demo/restart"
```

## E) Verify which commit is deployed

```bash
# Remote HEAD commit
git ls-remote https://huggingface.co/spaces/qbz506/pramana-nyaya-demo HEAD

# API metadata (contains current sha)
curl -H "Authorization: Bearer $HF_TOKEN" \
  "https://huggingface.co/api/spaces/qbz506/pramana-nyaya-demo"
```

## F) Test the Space via Gradio client

The Space exposes Gradio API endpoints. Use `gradio_client`:

```python
from gradio_client import Client

client = Client("qbz506/pramana-nyaya-demo")
print(client.view_api())

# Stage 1 base output
base = client.predict(
    "Stage 1 (DeepSeek 8B)",
    "Problem: If P then Q. If Q then R. P is true. What can you conclude?",
    "You are a Nyaya reasoning engine. Follow the exact output format provided.\n\n"
    "Use the exact section headers:\n"
    "## Samshaya (Doubt Analysis)\n"
    "## Pramana (Sources of Knowledge)\n"
    "## Pancha Avayava (5-Member Syllogism)\n"
    "## Tarka (Counterfactual Reasoning)\n"
    "## Hetvabhasa (Fallacy Check)\n"
    "## Nirnaya (Ascertainment)",
    api_name="/generate_base",
)

# Stage 1 tuned output
tuned = client.predict(
    "Stage 1 (DeepSeek 8B)",
    "Problem: If P then Q. If Q then R. P is true. What can you conclude?",
    "You are a Nyaya reasoning engine. Follow the exact output format provided.\n\n"
    "Use the exact section headers:\n"
    "## Samshaya (Doubt Analysis)\n"
    "## Pramana (Sources of Knowledge)\n"
    "## Pancha Avayava (5-Member Syllogism)\n"
    "## Tarka (Counterfactual Reasoning)\n"
    "## Hetvabhasa (Fallacy Check)\n"
    "## Nirnaya (Ascertainment)",
    api_name="/generate_tuned",
)
```

## G) ZeroGPU notes

ZeroGPU is time-constrained per task. If you see `"GPU task aborted"`:
- Split base and tuned generation into separate GPU tasks.
- Reduce `max_new_tokens` (Stage 1 defaults to 256).
- Disable caching for the large model.

These are implemented in the current Space app via:
- `STAGE1_MAX_NEW_TOKENS`
- `STAGE1_CACHE_MODELS`
