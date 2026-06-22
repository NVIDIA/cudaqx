"""Self-test for evaluate_metrics token summing."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluate_metrics import load_assertions, normalize_run, summarize


def _make_run(tmp_path, tokens_field):
    payload = {
        "agent": "test",
        "model": "test-model",
        "config": "with_skill",
        "responses": [
            {
                "id": "P1",
                "response": "cobyla requires gradients scipy L-BFGS-B jac",
                "tokens": tokens_field,
            }
        ],
    }
    path = tmp_path / "run.json"
    path.write_text(json.dumps(payload))
    return path


def _make_minimal_assertions(tmp_path):
    spec = {
        "P1": {
            "must_include": ["cobyla", "requires gradients", "scipy",
                             "L-BFGS-B", "jac"],
            "must_not_include": [],
        }
    }
    path = tmp_path / "assertions.json"
    path.write_text(json.dumps(spec))
    return path


def test_int_tokens(tmp_path):
    run_path = _make_run(tmp_path, 123)
    asserts_path = _make_minimal_assertions(tmp_path)
    summary = summarize(normalize_run(run_path),
                        load_assertions(asserts_path))
    assert summary["tokens_total"] == 123


def test_dict_tokens(tmp_path):
    run_path = _make_run(tmp_path, {"input": 100, "output": 50})
    asserts_path = _make_minimal_assertions(tmp_path)
    summary = summarize(normalize_run(run_path),
                        load_assertions(asserts_path))
    assert summary["tokens_total"] == 150


def test_missing_tokens(tmp_path):
    run_path = _make_run(tmp_path, None)
    asserts_path = _make_minimal_assertions(tmp_path)
    summary = summarize(normalize_run(run_path),
                        load_assertions(asserts_path))
    assert summary["tokens_total"] is None


def test_dict_all_nonnumeric(tmp_path):
    """Dict with no numeric values should yield None, matching missing/unknown."""
    run_path = _make_run(tmp_path, {"input": "abc", "output": None})
    asserts_path = _make_minimal_assertions(tmp_path)
    summary = summarize(normalize_run(run_path),
                        load_assertions(asserts_path))
    assert summary["tokens_total"] is None


def test_dict_all_numeric_zero(tmp_path):
    """A dict of numeric zeros is a legitimate 'zero tokens' run, not unparseable."""
    run_path = _make_run(tmp_path, {"input": 0, "output": 0})
    asserts_path = _make_minimal_assertions(tmp_path)
    summary = summarize(normalize_run(run_path),
                        load_assertions(asserts_path))
    assert summary["tokens_total"] == 0


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        test_int_tokens(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_dict_tokens(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_missing_tokens(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_dict_all_nonnumeric(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_dict_all_numeric_zero(Path(d))
    print("all token-sum tests passed")
