"""The writing model. Uses the Claude Code CLI headless (no tools, no web), so it needs no
API key beyond your Claude login. Tests and demos use FakeLLM, which is deterministic.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess


class LLMError(RuntimeError):
    pass


class ClaudeCLI:
    def __init__(self, model: str = "sonnet", binary: str | None = None, timeout: int = 180):
        self.model = model
        self.binary = binary or os.environ.get("CLAUDE_BIN") or shutil.which("claude") or os.path.expanduser("~/.local/bin/claude")
        self.timeout = timeout

    def complete(self, prompt: str) -> str:
        args = [self.binary, "-p", prompt, "--output-format", "json", "--model", self.model, "--tools", "",
                "--permission-mode", "dontAsk", "--strict-mcp-config", "--setting-sources", "project",
                "--disable-slash-commands", "--no-session-persistence"]
        try:
            done = subprocess.run(args, capture_output=True, text=True, timeout=self.timeout, stdin=subprocess.DEVNULL)
        except (OSError, subprocess.TimeoutExpired) as e:
            raise LLMError(str(e)) from e
        try:
            return (json.loads(done.stdout).get("result") or "").strip()
        except ValueError as e:
            raise LLMError((done.stderr or done.stdout).strip()[-200:] or f"claude exit {done.returncode}") from e


def extract_json(text: str):
    """The first JSON object/array in a model answer (models sometimes wrap it in prose or fences)."""
    text = re.sub(r"```(?:json)?", "", text)
    decoder = json.JSONDecoder()
    for i, c in enumerate(text):
        if c in "{[":
            try:
                return decoder.raw_decode(text, i)[0]
            except ValueError:
                continue
    raise LLMError("no JSON in model answer")


class FakeLLM:
    """Deterministic stand-in: answers with whatever `respond(prompt)` returns."""

    def __init__(self, respond):
        self.respond = respond
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.respond(prompt)
