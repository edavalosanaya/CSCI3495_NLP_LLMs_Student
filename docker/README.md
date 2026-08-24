# Course Environment Setup (Docker + Ollama)

Everything in this course runs in a single reproducible Docker image — **no GPU, no paid API keys, no cloud accounts**. Local LLMs run via **Ollama**.

## Prerequisites
- [Docker Desktop](https://docs.docker.com/get-docker/) (or Docker Engine + Compose v2) installed and running.
- ~10 GB free disk (image + a couple of small models).

## One-time setup
```bash
# From the repository root:
docker compose -f docker/docker-compose.yml build          # build the course image (~5–10 min first time)
docker compose -f docker/docker-compose.yml up -d ollama    # start the local LLM server

# Pull the small models the course uses (CPU-friendly):
docker compose -f docker/docker-compose.yml exec ollama ollama pull qwen2.5:0.5b
docker compose -f docker/docker-compose.yml exec ollama ollama pull llama3.2:1b
docker compose -f docker/docker-compose.yml exec ollama ollama pull nomic-embed-text   # embeddings for RAG
```

## Daily workflow
```bash
# Open a shell inside the course image (repo mounted at /workspace):
docker compose -f docker/docker-compose.yml run --rm course bash

# Or launch JupyterLab:
docker compose -f docker/docker-compose.yml run --rm --service-ports course \
    jupyter lab --ip=0.0.0.0 --no-browser --NotebookApp.token=''
# then open http://localhost:8888
```

Inside the container, the LLM server is reachable at `http://ollama:11434` (already set as `OLLAMA_HOST`). From your host it's `http://localhost:11434`.

## Quick check
```bash
docker compose -f docker/docker-compose.yml run --rm course python scripts/env_check.py
```
This verifies Python, the key libraries, and connectivity to Ollama.

## Using Ollama from Python
```python
import ollama
client = ollama.Client()  # reads OLLAMA_HOST from the environment
resp = client.chat(model="qwen2.5:0.5b",
                   messages=[{"role": "user", "content": "Say hello in one sentence."}])
print(resp["message"]["content"])
```

## Running locally without Docker (not recommended, but supported)
1. Install Python 3.12 and create a venv.
2. `pip install -r docker/requirements.txt` (install CPU torch first: `pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1`).
3. Install Ollama natively from <https://ollama.com> and `ollama serve`; set `OLLAMA_HOST=http://localhost:11434`.

## Troubleshooting
- **`permission denied ... /var/run/docker.sock`:** your user isn't in the `docker` group. Either prefix commands with `sudo`, or add yourself once: `sudo usermod -aG docker $USER` then log out/in (or `newgrp docker`).
- **`connection refused` to Ollama:** ensure the `ollama` service is up (`docker compose ... up -d ollama`) and the model is pulled.
- **Slow generation:** expected on CPU; the course uses sub-1B/1B models so responses take seconds, not minutes.
- **Out of disk:** `docker system prune` and remove unused models with `ollama rm <model>`.
