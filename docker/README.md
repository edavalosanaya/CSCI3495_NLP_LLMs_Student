# The local LLM server (Ollama)

Docker has exactly one job in this course: running a language model on your own
machine. **Python is not in Docker.** It is a single [uv](https://docs.astral.sh/uv/)
environment managed from the repository root, and that is where the notebooks run.

No GPU, no API keys, no cloud accounts, no paid anything.

## Prerequisites

- [Docker Desktop](https://docs.docker.com/get-docker/), or Docker Engine plus
  Compose v2 on Linux.
- About 5 GB of free disk for the server and a couple of small models.

On **Windows**, install Docker Desktop with the WSL 2 backend, which its
installer offers by default.

## One-time setup

From the repository root:

```bash
docker compose -f docker/docker-compose.yml up -d ollama
docker compose -f docker/docker-compose.yml exec ollama ollama pull qwen2.5:0.5b
```

That is the whole setup. The first pull is about 400 MB; everything after it is
instant.

`qwen2.5:0.5b` is the course default and is deliberately small: it runs on any
laptop CPU, and the labs are written so that a weak model still makes the point.
If your machine is comfortable, `qwen2.5:1.5b` gives noticeably better answers
and the labs work unchanged.

## Daily use

The server keeps running in the background once started, and restarts with
Docker. You do not interact with it directly: the notebooks talk to it over
`localhost:11434` through the `ollama` Python package, which is already in the
course environment.

```bash
docker compose -f docker/docker-compose.yml ps        # is it up?
docker compose -f docker/docker-compose.yml stop ollama
docker compose -f docker/docker-compose.yml up -d ollama
```

Check the whole environment, Python and server together, with:

```bash
uv run python scripts/env_check.py
```

Labs that need the model say so at the top, and every one of them falls back to
a canned reply if the server is not running, so a missing Ollama never stops you
finishing a notebook.

## Upgrading from the old setup

Earlier in the term the whole course ran inside a `csci3495-nlp` Docker image.
That image is no longer used and is three to five gigabytes of dead weight. To
reclaim the space, from the repository root:

```bash
bash scripts/cleanup_course_docker.sh
```

It shows you exactly what it will remove and asks before doing anything. It
removes only the course image and its model cache, and refuses to touch Ollama
or anything else on your machine.
