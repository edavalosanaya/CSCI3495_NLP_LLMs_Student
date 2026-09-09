#!/usr/bin/env python3
"""Reclaim the disk the old CSCI 3495 course image was using.

The course now runs on a uv environment instead of a Docker image, so the
csci3495-nlp image and its Hugging Face cache are dead weight, usually around
3 to 5 GB. Ollama STAYS in Docker and this script does not touch it.

Runs the same on macOS, Linux and Windows, which is why it is Python and not a
shell script: Windows has no bash, and PowerShell would mean a second copy to
keep in step with this one.

    python scripts/cleanup_course_docker.py          # show what would go, ask first
    python scripts/cleanup_course_docker.py --yes    # no prompt

This script is deliberately surgical. It never runs `docker system prune`,
never touches an image it did not name, and never removes the Ollama models
volume, which can hold several GB of downloaded models you would have to pull
again.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys

COURSE_IMAGE = "csci3495-nlp"
# named by docker-compose, with and without the compose project prefix
HF_VOLUMES = ["docker_hf_cache", "hf_cache"]
KEEP_PATTERN = "ollama"  # never removed, whatever happens


def docker(*args: str) -> str:
    """Run a docker command and return its stdout, empty on any failure."""
    try:
        out = subprocess.run(
            ["docker", *args],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


def lines(text: str) -> list[str]:
    return [ln for ln in text.splitlines() if ln.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-y", "--yes", action="store_true", help="do not prompt")
    args = ap.parse_args()

    if shutil.which("docker") is None:
        print("Docker is not installed. Nothing to clean up.")
        return 0
    if subprocess.run(
        ["docker", "info"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode != 0:
        print("Docker is installed but not running. Start Docker Desktop and re-run.")
        return 1

    print("Looking for the old course image and its cache...")
    print()

    # --- what we would remove ------------------------------------------------
    containers = lines(docker(
        "ps", "-a",
        "--filter", f"ancestor={COURSE_IMAGE}:latest",
        "--format", "{{.ID}} {{.Names}}",
    ))
    images = lines(docker(
        "image", "ls", COURSE_IMAGE,
        "--format", "{{.ID}} {{.Repository}}:{{.Tag}} {{.Size}}",
    ))
    volumes = [
        v for v in HF_VOLUMES
        if subprocess.run(
            ["docker", "volume", "inspect", v],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode == 0
    ]

    if not containers and not images and not volumes:
        print("Nothing to do: the course image is already gone.")
        print("(If you are looking for free space, this script only removes the course")
        print(" image. Everything else on your machine is left alone on purpose.)")
        return 0

    print("WILL REMOVE")
    for c in containers:
        print(f"   container  {c}")
    for i in images:
        print(f"   image      {i}")
    for v in volumes:
        print(f"   volume     {v}")
    print()
    print("WILL KEEP (Ollama is still used by the course)")
    for i in lines(docker("image", "ls", "--format", "{{.Repository}}:{{.Tag}} {{.Size}}")):
        if KEEP_PATTERN in i.lower():
            print(f"   image      {i}")
    for v in lines(docker("volume", "ls", "--format", "{{.Name}}")):
        if KEEP_PATTERN in v.lower():
            print(f"   volume     {v}")
    print("   ...and every other container, image and volume on this machine.")
    print()

    if not args.yes:
        try:
            reply = input("Remove the items listed under WILL REMOVE? [y/N] ")
        except EOFError:
            reply = ""
        if reply.strip().lower() not in {"y", "yes"}:
            print("Cancelled. Nothing was removed.")
            return 0

    # --- do it ---------------------------------------------------------------
    if containers:
        ids = [c.split()[0] for c in containers]
        subprocess.run(["docker", "rm", "-f", *ids],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        print(f"removed {len(ids)} container(s)")

    if images:
        # By repository:tag, never by a bare ID, so a shared layer cannot take an
        # unrelated image with it.
        for tag in (i.split()[1] for i in images):
            subprocess.run(["docker", "image", "rm", tag],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        print(f"removed the {COURSE_IMAGE} image")

    for v in volumes:
        if KEEP_PATTERN in v.lower():
            print(f"refusing to remove {v} (matches '{KEEP_PATTERN}')")
            continue
        removed = subprocess.run(["docker", "volume", "rm", v],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                 check=False).returncode == 0
        if removed:
            print(f"removed volume {v}")

    print()
    print("Done. Ollama and everything unrelated to this course were left alone.")
    print("Your Python environment is now managed by uv:  uv sync")
    return 0


if __name__ == "__main__":
    sys.exit(main())
