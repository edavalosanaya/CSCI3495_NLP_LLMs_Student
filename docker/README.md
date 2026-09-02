# Course Environment Setup (Docker + Ollama)

Everything in this course runs in a single reproducible Docker image, **no GPU, no paid API keys, no cloud accounts**. Local LLMs run via **Ollama**.

## Prerequisites
- [Docker Desktop](https://docs.docker.com/get-docker/) (or Docker Engine + Compose v2) installed and running.
- ~10 GB free disk (image + a couple of small models).

**Every command in this course works on macOS, Linux and Windows**, because the
work happens inside the Linux container, not on your machine. Type them in
Terminal (macOS, Linux) or in PowerShell or Command Prompt (Windows). The one
thing that differs between systems is how you make a shortcut for a long
command, and each handout spells out all three forms.

On **Windows**, install Docker Desktop with the WSL 2 backend (its installer
offers this and it is the default). Keep the repository on your `C:` drive
rather than a network drive, or the container will not see your edits. Use
forward slashes in the commands exactly as printed: they are paths inside the
container, not Windows paths.

## One-time setup
```bash
# From the repository root:
docker compose -f docker/docker-compose.yml build          # build the course image (~5-10 min first time)
docker compose -f docker/docker-compose.yml up -d ollama    # start the local LLM server

# Pull the small models the course uses (CPU-friendly):
docker compose -f docker/docker-compose.yml exec ollama ollama pull qwen2.5:0.5b
docker compose -f docker/docker-compose.yml exec ollama ollama pull llama3.2:1b
docker compose -f docker/docker-compose.yml exec ollama ollama pull nomic-embed-text   # embeddings for RAG
```

## Daily workflow

**Work inside the container.** Each lab's README opens with one command that
drops you into a shell already sitting in that lab's folder, so every command
after it is short: `pytest -k step1 -q`, `python text_tools.py`. You only paste
the long line once per session.

```bash
# The shape of it (each lab README gives you its own copy):
docker compose -f docker/docker-compose.yml run --rm --no-deps -w /workspace/weeks/week-01/class-02/exercise course bash

# Or a plain shell at the repository root:
docker compose -f docker/docker-compose.yml run --rm course bash

# Or launch JupyterLab:
docker compose -f docker/docker-compose.yml run --rm --service-ports course jupyter lab --ip=0.0.0.0 --no-browser --NotebookApp.token=""
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
- **`permission denied ... /var/run/docker.sock` (Linux):** your user isn't in the `docker` group. Either prefix commands with `sudo`, or add yourself once: `sudo usermod -aG docker $USER` then log out/in (or `newgrp docker`).
- **`docker: command not found`:** Docker Desktop is installed but not running. Start it and wait for the whale icon to stop animating, then try again.
- **Windows: the container starts but your edits do nothing.** The repository is probably on a network drive or outside your user folder. Move it under `C:\Users\<you>\` and re-run.
- **Windows: `docker compose` complains about the file path.** Run the commands from the repository root, the folder holding `docker/`, not from inside `docker/`.
- **`connection refused` to Ollama:** ensure the `ollama` service is up (`docker compose ... up -d ollama`) and the model is pulled.
- **Slow generation:** expected on CPU; the course uses sub-1B/1B models so responses take seconds, not minutes.
- **Out of disk:** `docker system prune` and remove unused models with `ollama rm <model>`.
