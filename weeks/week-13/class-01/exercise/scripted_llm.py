"""Deterministic fake "LLMs" for testing the W13 agent WITHOUT Ollama.
(Provided for you, no edits needed.)

  * ScriptedLLM([...]) , replays a fixed list of outputs (per call).
  * ReflexiveLLM(...)  , switches from a "before" to an "after" script once a
                          reflection note appears in the transcript, so you can
                          test that reflection actually changes behavior.
"""
from __future__ import annotations

from typing import Callable


class ScriptedLLM:
    def __init__(self, script: list[str]):
        self.script = list(script)
        self.calls: list[str] = []

    def __call__(self, transcript: str) -> str:
        self.calls.append(transcript)
        if not self.script:
            return "Thought: out of script.\nAction: finish[I don't know]"
        return self.script.pop(0)


class ReflexiveLLM:
    def __init__(self, before: list[str], after: list[str], marker: str = "Lessons from previous attempts"):
        self.before = list(before)
        self.after = list(after)
        self.marker = marker
        self.calls: list[str] = []

    def __call__(self, transcript: str) -> str:
        self.calls.append(transcript)
        script = self.after if self.marker in transcript else self.before
        if not script:
            return "Thought: stuck.\nAction: finish[wrong]"
        return script.pop(0)


def const(text: str) -> Callable[[str], str]:
    return lambda _prompt: text
