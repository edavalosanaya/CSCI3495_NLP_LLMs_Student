#!/usr/bin/env python3
"""Environment sanity check for CSCI 3495.

Verifies the core libraries import and that a local Ollama server is reachable.
Exit code 0 = environment OK (Ollama optional but reported).
"""
from __future__ import annotations
import importlib
import os
import sys

CORE = [
    "numpy", "scipy", "pandas", "sklearn", "matplotlib",
    "nltk", "regex", "torch", "transformers", "datasets",
    "sentence_transformers", "peft", "faiss", "ollama",
]


def check_imports() -> bool:
    ok = True
    for mod in CORE:
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, "__version__", "?")
            print(f"  [ok]  {mod:<22} {ver}")
        except Exception as e:  # noqa: BLE001
            ok = False
            print(f"  [FAIL] {mod:<22} {e}")
    return ok


def check_ollama() -> bool:
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    try:
        import ollama
        client = ollama.Client(host=host)
        models = client.list().get("models", [])
        names = [m.get("model", m.get("name", "?")) for m in models]
        print(f"  [ok]  Ollama reachable at {host}; models: {names or '(none pulled yet)'}")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] Ollama not reachable at {host}: {e}")
        print("         Start it with: docker compose -f docker/docker-compose.yml up -d ollama")
        return False


def main() -> int:
    print(f"Python {sys.version.split()[0]}")
    print("Core libraries:")
    imports_ok = check_imports()
    print("Ollama:")
    check_ollama()  # optional, not fatal
    if not imports_ok:
        print("\nENV CHECK FAILED: some core libraries are missing.")
        return 1
    print("\nENV CHECK OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
