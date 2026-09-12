#!/usr/bin/env python3
"""Force-update the course repo to match the instructor's latest version.

Discards every local change to the course material (labs, decks, solutions,
scripts) and pulls the newest version. Your own work is protected: anything
under the folders listed in KEEP is copied aside first, then put back, so
homework and project files survive untouched. Files you created yourself are
untracked by git and are never touched.

A dated copy of the protected folders is left in .backups/ every run.

Usage:  python scripts/update_course.py
Exit code 0 = repo is up to date.
"""
from __future__ import annotations
import datetime as dt
import shutil
import subprocess
import sys
from pathlib import Path

# Folders whose contents are yours and are never discarded.
KEEP = ["homeworks", "project"]

REPO = Path(__file__).resolve().parent.parent
BACKUPS = REPO / ".backups"


def git(*args: str, capture: bool = False) -> str:
    """Run a git command in the repo, exiting on failure."""
    proc = subprocess.run(
        ["git", *args],
        cwd=REPO,
        text=True,
        capture_output=capture,
    )
    if proc.returncode != 0:
        if capture and proc.stderr:
            print(proc.stderr.strip(), file=sys.stderr)
        sys.exit(f"[FAIL] git {' '.join(args)} failed.")
    return (proc.stdout or "").strip()


def default_branch() -> str:
    """The branch origin points at, falling back to main."""
    try:
        ref = git("symbolic-ref", "--quiet", "refs/remotes/origin/HEAD", capture=True)
        return ref.rsplit("/", 1)[-1] or "main"
    except SystemExit:
        return "main"


def backup(stamp: str) -> Path:
    dest = BACKUPS / stamp
    for name in KEEP:
        src = REPO / name
        if src.is_dir():
            shutil.copytree(src, dest / name)
            print(f"  [ok]  saved {name}/")
    return dest


def restore(saved: Path) -> None:
    for name in KEEP:
        src = saved / name
        if src.is_dir():
            shutil.copytree(src, REPO / name, dirs_exist_ok=True)
            print(f"  [ok]  restored {name}/")


def main() -> int:
    if not (REPO / ".git").exists():
        sys.exit(f"[FAIL] {REPO} is not a git repository.")

    branch = default_branch()
    print(f"Updating {REPO.name} from origin/{branch}\n")

    stamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    print("Saving your work:")
    saved = backup(stamp)

    print("\nFetching the latest course material:")
    git("fetch", "origin")

    # Commits you made that the instructor does not have would be thrown away by
    # the reset below, so park them on a branch first. Nothing committed is lost.
    ahead = git("rev-list", "--count", f"origin/{branch}..HEAD", capture=True)
    if ahead != "0":
        saved_branch = f"my-work-{stamp}"
        git("branch", saved_branch)
        print(f"  [ok]  your {ahead} commit(s) saved on branch {saved_branch}")

    print("Discarding local changes to course material:")
    # Resets every course file to the instructor's version. Files you created
    # yourself and never committed are untracked, and are left where they are.
    git("reset", "--hard", f"origin/{branch}")

    print("\nPutting your work back:")
    restore(saved)

    print(f"\nDone. You are on the latest version of the course material.")
    print(f"A copy of your {', '.join(n + '/' for n in KEEP)} is in {saved.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
