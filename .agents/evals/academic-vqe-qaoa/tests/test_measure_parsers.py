#!/usr/bin/env python3
"""Unit tests for measure_tokens.py parsers against real CLI sample payloads.

These fixtures mirror the documented headless output of each agent CLI, so the
token extraction is verified without needing any CLI installed.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from measure_tokens import (  # noqa: E402
    parse_claude_json,
    parse_codex_jsonl,
    parse_cursor_json,
    record_meter,
)


def test_claude_real_usage():
    sample = json.dumps({
        "result": "answer text",
        "session_id": "s1",
        "total_cost_usd": 0.0079825,
        "usage": {"input_tokens": 3, "output_tokens": 6,
                  "cache_read_input_tokens": 15635},
    })
    text, usage, cost = parse_claude_json(sample)
    assert text == "answer text"
    assert usage == {"input_tokens": 3, "output_tokens": 6}
    assert cost == 0.0079825


def test_codex_jsonl_usage_and_text():
    sample = "\n".join([
        '{"type":"thread.started","thread_id":"t"}',
        '{"type":"turn.started"}',
        '{"type":"item.completed","item":{"id":"i","type":"agent_message",'
        '"text":"final answer"}}',
        '{"type":"turn.completed","usage":{"input_tokens":24763,'
        '"cached_input_tokens":24448,"output_tokens":122,'
        '"reasoning_output_tokens":0}}',
    ])
    text, usage, cost = parse_codex_jsonl(sample)
    assert text == "final answer"
    assert usage == {"input_tokens": 24763, "output_tokens": 122}
    assert cost is None


def test_codex_token_count_event_fallback():
    sample = "\n".join([
        '{"type":"item.completed","item":{"type":"agent_message","text":"x"}}',
        '{"type":"event_msg","payload":{"type":"token_count","info":'
        '{"total_token_usage":{"input_tokens":8408,"output_tokens":7}}}}',
    ])
    _, usage, _ = parse_codex_jsonl(sample)
    assert usage == {"input_tokens": 8408, "output_tokens": 7}


def test_cursor_with_usage():
    sample = json.dumps({
        "result": "hi", "chatId": "abc", "model": "gpt-5",
        "usage": {"input_tokens": 10, "output_tokens": 4},
    })
    text, usage, _ = parse_cursor_json(sample)
    assert text == "hi"
    assert usage == {"input_tokens": 10, "output_tokens": 4}


def test_cursor_without_usage_is_unavailable():
    sample = json.dumps({"result": "hi", "chatId": "x", "model": "m"})
    _, usage, _ = parse_cursor_json(sample)
    assert usage is None


def test_record_meter_totals_and_source():
    rec = record_meter("P5", "claude",
                        ("txt", {"input_tokens": 100, "output_tokens": 50}, 0.01))
    assert rec["tokens"] == 150
    assert rec["tokens_source"] == "meter"
    assert rec["cost_usd"] == 0.01


def test_record_meter_unavailable_when_no_usage():
    rec = record_meter("P5", "cursor", ("txt", None, None))
    assert rec["tokens"] is None
    assert rec["tokens_source"] == "unavailable"
    assert "cost_usd" not in rec


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  [ok] {fn.__name__}")
    print(f"{len(fns)} passed")
