# CSCI 3495 — NLP & LLMs course image (CPU-only, reproducible)
FROM python:3.12-slim-bookworm

# System deps: build tools for some wheels, git, curl
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/student/.cache/huggingface \
    NLTK_DATA=/usr/share/nltk_data \
    # By default talk to an Ollama running on the host or the compose service:
    OLLAMA_HOST=http://ollama:11434

# Install CPU-only PyTorch first (smaller, no CUDA), then the rest.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cpu torch==2.12.1 \
    && pip install -r /tmp/requirements.txt

# Pre-download small NLTK corpora used in early weeks
RUN python -m nltk.downloader -d /usr/share/nltk_data punkt punkt_tab stopwords wordnet

# Non-root user for safety; mount the repo at /workspace
RUN useradd -ms /bin/bash student \
    && mkdir -p /home/student/.cache/huggingface \
    && chown -R student:student /home/student/.cache
# Pre-creating the cache dir (owned by student) makes the named hf_cache volume
# inherit student ownership, so model downloads in homeworks are writable.
USER student
WORKDIR /workspace

CMD ["bash"]
