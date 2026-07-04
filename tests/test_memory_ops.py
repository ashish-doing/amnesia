"""
Unit tests for memory/memory_ops.py's pure, Cognee-free confidence-tracking
logic. Deliberately zero-dependency - no Cognee, no network, no API keys -
because this logic (improve_from_outcome, the confidence store) has to be
correct on its own before anything built on top of it (the memory graph
visualization, the drift-correction demo) can be trusted.

These tests use a temporary confidence store file, monkeypatched in, so they
never touch your real memory/confidence_store.json.
"""
import sys
import json
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from memory import memory_ops


def _with_temp_store(fn):
    """Run fn with memory_ops.CONFIDENCE_STORE pointed at a fresh temp file,
    restoring the original path afterward regardless of outcome."""
    original = memory_ops.CONFIDENCE_STORE
    with tempfile.TemporaryDirectory() as d:
        memory_ops.CONFIDENCE_STORE = Path(d) / "confidence_store.json"
        try:
            fn()
        finally:
            memory_ops.CONFIDENCE_STORE = original


def test_improve_from_outcome_creates_entry_if_missing():
    def run():
        memory_ops.improve_from_outcome("new_fact", was_correct=True, session=1)
        store = memory_ops._load_confidence()
        assert "new_fact" in store
        assert store["new_fact"]["confidence"] > 0.5  # started at 0.5, boosted by +0.2
    _with_temp_store(run)


def test_improve_from_outcome_increases_confidence_on_correct():
    def run():
        memory_ops.improve_from_outcome("fact_a", was_correct=True, session=1)
        first = memory_ops._load_confidence()["fact_a"]["confidence"]
        memory_ops.improve_from_outcome("fact_a", was_correct=True, session=2)
        second = memory_ops._load_confidence()["fact_a"]["confidence"]
        assert second > first
    _with_temp_store(run)


def test_improve_from_outcome_decreases_confidence_on_incorrect():
    def run():
        memory_ops.improve_from_outcome("fact_b", was_correct=True, session=1)
        before = memory_ops._load_confidence()["fact_b"]["confidence"]
        memory_ops.improve_from_outcome("fact_b", was_correct=False, session=2)
        after = memory_ops._load_confidence()["fact_b"]["confidence"]
        assert after < before
    _with_temp_store(run)


def test_improve_from_outcome_confidence_never_exceeds_one():
    def run():
        for _ in range(20):
            memory_ops.improve_from_outcome("fact_c", was_correct=True, session=1)
        assert memory_ops._load_confidence()["fact_c"]["confidence"] <= 1.0
    _with_temp_store(run)


def test_improve_from_outcome_confidence_never_below_zero():
    def run():
        memory_ops.improve_from_outcome("fact_d", was_correct=True, session=1)
        for _ in range(20):
            memory_ops.improve_from_outcome("fact_d", was_correct=False, session=1)
        assert memory_ops._load_confidence()["fact_d"]["confidence"] >= 0.0
    _with_temp_store(run)


def test_load_confidence_returns_empty_dict_when_no_file():
    def run():
        assert memory_ops._load_confidence() == {}
    _with_temp_store(run)


def test_save_and_load_round_trip():
    def run():
        memory_ops._save_confidence({"x": {"confidence": 0.7, "text": "hello", "last_confirmed_session": 3}})
        loaded = memory_ops._load_confidence()
        assert loaded["x"]["confidence"] == 0.7
        assert loaded["x"]["text"] == "hello"
    _with_temp_store(run)


if __name__ == "__main__":
    import traceback
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
            passed += 1
        except AssertionError:
            print(f"FAIL {t.__name__}")
            traceback.print_exc()
    print(f"\n{passed}/{len(tests)} passed")
