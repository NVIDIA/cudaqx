#!/usr/bin/env python3
"""Record a workshop run's REAL token usage into a run-JSON the scorer reads.

Workshop asset: run the same prompt with and without the skill, capture the
*actual* token counts the agent's API meter reports, and produce a run-JSON
that ``evaluate_metrics.py`` consumes.

This is a real recorder, not a guess. It invokes an installed agent CLI in
headless mode and parses the token usage out of its structured output:

* Claude Code  ``claude -p "<prompt>" --output-format json``
                 -> ``.usage.input_tokens`` / ``.usage.output_tokens`` /
                    ``.total_cost_usd``
* Codex        ``codex exec --json "<prompt>"``
                 -> last ``turn.completed.usage`` event
* Cursor       ``cursor-agent -p --output-format json "<prompt>"``
                 -> ``.usage`` (present in recent CLI versions)

Manual entry is the *only* fallback, used when no runtime is installed. Records
written that way are tagged ``"tokens_source": "manual"`` so they can never be
mistaken for meter readings.

Output schema (consumed by evaluate_metrics.py)::

    {"agent": "claude-code", "config": "with_skill", "responses": [
        {"id": "P5", "response": "...", "tokens": 1234,
         "usage": {"input_tokens": 1000, "output_tokens": 234},
         "cost_usd": 0.01, "tokens_source": "meter", "runtime": "claude"}]}

Run ``python measure_tokens.py --selftest`` to verify the parsers against real
sample payloads without needing any CLI installed.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parent
PROMPTS = ROOT / "prompts.json"

# (binary candidates, runtime key, agent label)
RUNTIMES = (
    (("claude",), "claude", "claude-code"),
    (("codex",), "codex", "codex"),
    (("cursor-agent", "agent"), "cursor", "cursor"),
)

Usage = Optional[dict[str, int]]
Parsed = tuple[str, Usage, Optional[float]]


# --------------------------------------------------------------------------- #
# Pure parsers (unit-testable without any CLI installed)
# --------------------------------------------------------------------------- #
def parse_claude_json(stdout: str) -> Parsed:
    """Parse ``claude -p --output-format json`` output."""
    obj = json.loads(stdout)
    text = str(obj.get("result", ""))
    usage = None
    raw = obj.get("usage")
    if isinstance(raw, dict):
        usage = {
            "input_tokens": int(raw.get("input_tokens", 0)),
            "output_tokens": int(raw.get("output_tokens", 0)),
        }
    cost = obj.get("total_cost_usd")
    cost = float(cost) if isinstance(cost, (int, float)) else None
    return text, usage, cost


def parse_codex_jsonl(stdout: str) -> Parsed:
    """Parse ``codex exec --json`` JSON-lines output.

    Text comes from the last ``agent_message`` item; usage from the last
    ``turn.completed`` event (or a ``token_count`` event_msg as a fallback).
    """
    text = ""
    usage: Usage = None
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            evt = json.loads(line)
        except json.JSONDecodeError:
            continue
        etype = evt.get("type")
        if etype == "item.completed":
            item = evt.get("item", {})
            if item.get("type") == "agent_message" and item.get("text"):
                text = str(item["text"])
        elif etype == "turn.completed":
            u = evt.get("usage", {})
            if isinstance(u, dict):
                usage = {
                    "input_tokens": int(u.get("input_tokens", 0)),
                    "output_tokens": int(u.get("output_tokens", 0)),
                }
        elif etype == "event_msg":
            payload = evt.get("payload", {})
            if payload.get("type") == "token_count":
                tot = payload.get("info", {}).get("total_token_usage", {})
                if isinstance(tot, dict) and usage is None:
                    usage = {
                        "input_tokens": int(tot.get("input_tokens", 0)),
                        "output_tokens": int(tot.get("output_tokens", 0)),
                    }
    return text, usage, None


def parse_cursor_json(stdout: str) -> Parsed:
    """Parse ``cursor-agent -p --output-format json`` output."""
    obj = json.loads(stdout)
    text = str(obj.get("result", ""))
    usage = None
    raw = obj.get("usage")
    if isinstance(raw, dict):
        in_tok = raw.get("input_tokens", raw.get("inputTokens", 0))
        out_tok = raw.get("output_tokens", raw.get("outputTokens", 0))
        usage = {"input_tokens": int(in_tok), "output_tokens": int(out_tok)}
    return text, usage, None


PARSERS = {
    "claude": parse_claude_json,
    "codex": parse_codex_jsonl,
    "cursor": parse_cursor_json,
}


def build_command(runtime: str, binary: str, prompt: str,
                  model: Optional[str], extra: list[str]) -> list[str]:
    if runtime == "claude":
        cmd = [binary, "-p", prompt, "--output-format", "json"]
        if model:
            cmd += ["--model", model]
    elif runtime == "codex":
        cmd = [binary, "exec", "--json", "--skip-git-repo-check"]
        if model:
            cmd += ["-m", model]
        cmd += [prompt]
    elif runtime == "cursor":
        cmd = [binary, "-p", "--output-format", "json"]
        if model:
            cmd += ["-m", model]
        cmd += [prompt]
    else:
        raise ValueError(f"no command for runtime {runtime!r}")
    return cmd + extra


# --------------------------------------------------------------------------- #
# Runtime detection + invocation
# --------------------------------------------------------------------------- #
def detect_runtime() -> tuple[str, Optional[str], str]:
    """Return (runtime_key, binary, agent_label); ('manual', None, 'manual')."""
    for binaries, runtime, label in RUNTIMES:
        for binary in binaries:
            if shutil.which(binary):
                return runtime, binary, label
    return "manual", None, "manual"


def resolve_binary(runtime: str) -> Optional[str]:
    for binaries, key, _ in RUNTIMES:
        if key == runtime:
            for binary in binaries:
                if shutil.which(binary):
                    return binary
    return None


def invoke(runtime: str, binary: str, prompt: str, model: Optional[str],
           extra: list[str], timeout: int, dry_run: bool) -> Parsed:
    cmd = build_command(runtime, binary, prompt, model, extra)
    if dry_run:
        print("DRY RUN:", " ".join(repr(c) if " " in c else c for c in cmd))
        return "", None, None
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"{runtime} exited {proc.returncode}")
    return PARSERS[runtime](proc.stdout)


# --------------------------------------------------------------------------- #
# Manual fallback
# --------------------------------------------------------------------------- #
def ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    try:
        value = input(f"{prompt}{suffix}: ").strip()
    except EOFError:
        return default
    return value or default


def read_response() -> str:
    print("Paste the agent's response, then a line with only END:")
    lines: list[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == "END":
            break
        lines.append(line)
    return "\n".join(lines).strip()


def ask_int(prompt: str) -> int:
    raw = ask(prompt, "0")
    try:
        return int(raw)
    except ValueError:
        return 0


def record_manual(pid: str, prompt: str) -> dict[str, Any]:
    print(f"\n--- {pid} (MANUAL: no agent runtime detected) ---\n{prompt}\n")
    in_tok = ask_int("Input tokens shown by your agent UI")
    out_tok = ask_int("Output tokens shown by your agent UI")
    response = read_response()
    return {
        "id": pid,
        "response": response,
        "tokens": in_tok + out_tok,
        "usage": {"input_tokens": in_tok, "output_tokens": out_tok},
        "tokens_source": "manual",
        "runtime": "manual",
    }


def record_meter(pid: str, runtime: str, parsed: Parsed) -> dict[str, Any]:
    text, usage, cost = parsed
    total = None
    if usage is not None:
        total = int(usage.get("input_tokens", 0) + usage.get("output_tokens", 0))
    rec: dict[str, Any] = {
        "id": pid,
        "response": text,
        "tokens": total,
        "usage": usage,
        "tokens_source": "meter" if usage is not None else "unavailable",
        "runtime": runtime,
    }
    if cost is not None:
        rec["cost_usd"] = cost
    return rec


# --------------------------------------------------------------------------- #
def load_prompts() -> dict[str, str]:
    data = json.loads(PROMPTS.read_text())
    return {p["id"]: p["prompt"] for p in data}


def selftest() -> int:
    """Verify the parsers against real sample payloads (no CLI needed)."""
    claude_sample = json.dumps({
        "result": "Hello from Claude.",
        "session_id": "s1",
        "total_cost_usd": 0.0079825,
        "usage": {"input_tokens": 3, "output_tokens": 6,
                  "cache_read_input_tokens": 15635},
    })
    codex_sample = "\n".join([
        '{"type":"thread.started","thread_id":"t"}',
        '{"type":"turn.started"}',
        '{"type":"item.completed","item":{"id":"i","type":"agent_message",'
        '"text":"Repo contains docs."}}',
        '{"type":"turn.completed","usage":{"input_tokens":24763,'
        '"cached_input_tokens":24448,"output_tokens":122,'
        '"reasoning_output_tokens":0}}',
    ])
    cursor_sample = json.dumps({
        "result": "Hello!", "chatId": "abc", "model": "gpt-5",
        "usage": {"input_tokens": 10, "output_tokens": 4},
    })
    cursor_no_usage = json.dumps({"result": "Hi", "chatId": "x", "model": "m"})

    checks = []
    t, u, c = parse_claude_json(claude_sample)
    checks.append((t == "Hello from Claude." and u == {
        "input_tokens": 3, "output_tokens": 6} and c == 0.0079825,
        "claude"))
    t, u, c = parse_codex_jsonl(codex_sample)
    checks.append((t == "Repo contains docs." and u == {
        "input_tokens": 24763, "output_tokens": 122} and c is None,
        "codex"))
    t, u, c = parse_cursor_json(cursor_sample)
    checks.append((t == "Hello!" and u == {
        "input_tokens": 10, "output_tokens": 4}, "cursor"))
    t, u, c = parse_cursor_json(cursor_no_usage)
    checks.append((t == "Hi" and u is None, "cursor-no-usage"))

    ok = True
    for passed, name in checks:
        print(f"  [{'ok' if passed else 'FAIL'}] {name}")
        ok = ok and passed
    print("selftest:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true",
                        help="Verify parsers on sample payloads and exit.")
    parser.add_argument("--prompt-ids", default=None,
                        help="Comma-separated prompt ids (default: ask).")
    parser.add_argument("--config", default=None,
                        help="with_skill or without_skill (label only).")
    parser.add_argument("--runtime", default="auto",
                        choices=["auto", "claude", "codex", "cursor", "manual"])
    parser.add_argument("--model", default=None, help="Model id (optional).")
    parser.add_argument("--agent", default=None, help="Agent label override.")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output run-JSON path.")
    parser.add_argument("--timeout", type=int, default=600,
                        help="Per-prompt CLI timeout (seconds).")
    parser.add_argument("--extra-args", default="",
                        help="Extra args appended to the CLI invocation.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the CLI command(s) without running.")
    args = parser.parse_args()

    if args.selftest:
        return selftest()

    if args.runtime == "auto":
        runtime, binary, label = detect_runtime()
    else:
        runtime = args.runtime
        binary = None if runtime == "manual" else resolve_binary(runtime)
        label = {"claude": "claude-code"}.get(runtime, runtime)
        if runtime != "manual" and binary is None:
            if args.dry_run:
                binary = next(bins[0] for bins, key, _ in RUNTIMES
                              if key == runtime)  # candidate name for preview
            else:
                raise SystemExit(f"runtime '{runtime}' requested but its CLI is "
                                 f"not on PATH")

    prompts = load_prompts()
    print(f"Runtime: {runtime}" + (f" ({binary})" if binary else ""))
    if runtime == "manual":
        print("No agent CLI detected -> MANUAL entry. Numbers you type are "
              "tagged tokens_source=manual (not meter readings).")

    agent = args.agent or label
    config = args.config or ask("Config (with_skill/without_skill)",
                                "with_skill")
    out = args.out or (ROOT / "runs" / f"{config}.json")
    extra = args.extra_args.split() if args.extra_args else []

    if args.prompt_ids:
        ids = [p.strip() for p in args.prompt_ids.split(",") if p.strip()]
    else:
        print(f"Known prompt ids: {', '.join(prompts)}")
        ids = []
        while True:
            pid = ask("Prompt id (blank to finish)")
            if not pid:
                break
            ids.append(pid)

    responses: list[dict[str, Any]] = []
    for pid in ids:
        if pid not in prompts:
            print(f"  unknown id '{pid}', skipping.")
            continue
        if runtime == "manual":
            responses.append(record_manual(pid, prompts[pid]))
            continue
        print(f"\n--- {pid} via {runtime} ---")
        parsed = invoke(runtime, binary, prompts[pid], args.model, extra,
                        args.timeout, args.dry_run)
        if args.dry_run:
            continue
        rec = record_meter(pid, runtime, parsed)
        src = rec["tokens_source"]
        tok = rec["tokens"]
        print(f"  tokens={tok} ({src})"
              + (f", cost_usd={rec['cost_usd']}" if "cost_usd" in rec else ""))
        responses.append(rec)

    if args.dry_run or not responses:
        if not responses and not args.dry_run:
            print("No responses recorded; nothing written.")
        return 0

    payload = {"agent": agent, "config": config, "model": args.model or "",
               "responses": responses}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {len(responses)} response(s) to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
