"""Tool layer tests: safe arithmetic, conversions, notes search, pre-router, router.

The router contract matters for the demo: malformed tool JSON must never crash
and must come back as plain text (the callers suppress it from speech).
"""

import pytest

from snap import config
from snap.tools import calculate, convert, pre_route, route, search_notes

# --- calculate ---------------------------------------------------------------

def test_calculate_basic_arithmetic() -> None:
    assert calculate("2+3*4") == 14.0
    assert calculate("(2+3)*4") == 20.0
    assert calculate("10/4") == 2.5
    assert calculate("7%3") == 1.0


def test_calculate_pow_and_unary_minus() -> None:
    assert calculate("2**10") == 1024.0
    assert calculate("-5") == -5.0


def test_calculate_rejects_non_arithmetic() -> None:
    with pytest.raises(ValueError):
        calculate("__import__('os')")  # Call node — not in the whitelist
    with pytest.raises(ValueError):
        calculate("'string'")  # non-numeric constant
    with pytest.raises(SyntaxError):
        calculate("import os")  # not an expression


# --- convert -----------------------------------------------------------------

def test_convert_distance_and_weight() -> None:
    assert convert(5, "km", "mi") == pytest.approx(3.106855)
    assert convert(12, "kg", "lb") == pytest.approx(26.45544)


def test_convert_temperature() -> None:
    assert convert(100, "c", "f") == pytest.approx(212.0)
    assert convert(32, "f", "c") == pytest.approx(0.0)


def test_convert_case_insensitive_and_rejects_unknown() -> None:
    assert convert(1, "KM", "Mi") == pytest.approx(0.621371)
    with pytest.raises(ValueError):
        convert(1, "km", "banana")


# --- search_notes ------------------------------------------------------------

def test_search_notes_ranks_best_file(tmp_path) -> None:
    (tmp_path / "a.md").write_text("Plants use photosynthesis. More photosynthesis. Even more photosynthesis.", "utf-8")
    (tmp_path / "b.md").write_text("photosynthesis mentioned once.", "utf-8")
    result = search_notes("what is photosynthesis", notes_dir=tmp_path)
    assert result.startswith("From your note 'a':")


def test_search_notes_strips_markdown_heading(tmp_path) -> None:
    (tmp_path / "c.md").write_text("## Evaporation basics — evaporation is water turning to vapor.", "utf-8")
    result = search_notes("evaporation", notes_dir=tmp_path)
    assert result.startswith("From your note 'c': Evaporation basics")
    assert "#" not in result


def test_search_notes_no_match(tmp_path) -> None:
    (tmp_path / "a.md").write_text("nothing relevant here", "utf-8")
    assert search_notes("quantum entanglement", notes_dir=tmp_path) == "No notes matched 'quantum entanglement'."


# --- pre_router (deterministic, zero-token answers) ---------------------------

def test_pre_route_percent() -> None:
    assert pre_route("What is 15% of 2400?") == "That's 360."
    assert pre_route("15 percent of 2,400") == "That's 360."


def test_pre_route_square_root() -> None:
    assert pre_route("What is the square root of 1764?") == "The square root of 1764 is 42."


def test_pre_route_conversions() -> None:
    assert pre_route("Convert 5 km to miles") == "That's 3.10685 mi."
    assert pre_route("12 kg in pounds") == "That's 26.4554 lb."


def test_pre_route_binary_arithmetic() -> None:
    assert pre_route("2400 * 12") == "That's 28800."
    assert pre_route("9 divided by 2") == "That's 4.5."
    assert pre_route("12 plus 3") == "That's 15."


def test_pre_route_ignores_conversation() -> None:
    assert pre_route("What's the weather like?") is None
    assert pre_route("Tell me about photosynthesis") is None


# --- router ------------------------------------------------------------------

def test_route_plain_text_passthrough() -> None:
    assert route("Hello there") == "Hello there"


def test_route_empty_string() -> None:
    assert route("") == "…"


def test_route_say_decision() -> None:
    assert route('{"say": "Hello!"}') == "Hello!"


def test_route_calculate_tool() -> None:
    out = route('{"tool": "calculate", "args": {"expression": "2400*0.15"}}')
    assert out == "That's 360."


def test_route_convert_tool() -> None:
    out = route('{"tool": "convert", "args": {"value": 5, "from_unit": "km", "to_unit": "mi"}}')
    assert out == "That's 3.10685 mi."


def test_route_search_notes_tool(tmp_path, monkeypatch) -> None:
    (tmp_path / "n.md").write_text("The water cycle moves water around Earth.", "utf-8")
    monkeypatch.setattr(config, "NOTES_DIR", tmp_path)
    out = route('{"tool": "search_notes", "args": {"query": "water cycle"}}')
    assert out.startswith("From your note 'n':")


def test_route_json_wrapped_in_prose() -> None:
    out = route('Sure! {"tool": "calculate", "args": {"expression": "1+1"}}')
    assert out == "That's 2."


def test_route_malformed_json_returns_raw_text() -> None:
    raw = '{"tool": broken'
    assert route(raw) == raw  # never raises — callers decide what to do with it


def test_route_braces_but_not_json() -> None:
    raw = "use {curly} braces"
    assert route(raw) == raw


def test_route_unknown_tool_spoken_error() -> None:
    out = route('{"tool": "fly"}')
    assert out.startswith("Sorry, that tool failed:")


def test_route_tool_failure_spoken_error() -> None:
    out = route('{"tool": "convert", "args": {"value": 1, "from_unit": "km", "to_unit": "banana"}}')
    assert out.startswith("Sorry, that tool failed:")
