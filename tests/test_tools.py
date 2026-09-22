"""Tool layer tests: safe arithmetic, conversions, notes search, pre-router, router.

The router contract matters for the demo: malformed tool JSON must never crash
and must come back as plain text (the callers suppress it from speech).
"""

import pytest

from snap import config
from snap.tools import (
    TOOL_CALL_BLOCK,
    TOOLS,
    calculate,
    convert,
    dispatch,
    execute_round,
    openai_schemas,
    parse_tool_calls,
    pre_route,
    route,
    schemas_block,
    search_notes,
    speakable,
)

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


# --- registry + Hermes-style protocol ------------------------------------------


def test_registry_has_all_tools() -> None:
    assert set(TOOLS) == {"calculate", "convert", "search_notes", "add_note", "get_datetime"}


def test_schemas_block_renders_valid_json_schemas() -> None:
    block = schemas_block()
    assert "<tools>" in block and "</tools>" in block
    for name in TOOLS:
        assert f'"name": "{name}"' in block
    # every rendered schema must be valid JSON on its own line-group
    for schema in openai_schemas():
        assert schema["type"] == "function"
        assert "parameters" in schema["function"]


def test_parse_tool_calls_extracts_tags() -> None:
    raw = 'Sure. <tool_call>\n{"name": "calculate", "arguments": {"expression": "1+1"}}\n</tool_call>'
    assert parse_tool_calls(raw) == [("calculate", {"expression": "1+1"})]


def test_parse_tool_calls_skips_malformed_and_unterminated() -> None:
    assert parse_tool_calls('<tool_call>{"name": "calculate", "args": broken</tool_call>') == []
    assert parse_tool_calls('<tool_call>{"name": "calculate", "arguments": {}}') == []  # unterminated
    assert parse_tool_calls("no tags here") == []


def test_parse_tool_calls_legacy_json_fallback() -> None:
    raw = '{"tool": "convert", "args": {"value": 5, "from_unit": "km", "to_unit": "mi"}}'
    assert parse_tool_calls(raw) == [
        ("convert", {"value": 5, "from_unit": "km", "to_unit": "mi"})
    ]


def test_dispatch_success_and_unknown_tool() -> None:
    assert dispatch("calculate", {"expression": "2+2"}) == "4.0"
    err = dispatch("fly", {})
    assert err.startswith("There was an error when executing the function: fly")
    assert "calculate" in err  # tells the model which tools exist (recovery)


def test_dispatch_handler_error_returns_model_recoverable_message() -> None:
    err = dispatch("convert", {"value": 1, "from_unit": "km", "to_unit": "banana"})
    assert "There was an error when executing the function: convert" in err
    assert "unsupported unit pair" in err


def test_execute_round_plain_text_is_none() -> None:
    assert execute_round("Just a plain answer.") is None


def test_execute_round_templated_tool_speaks_fastpath() -> None:
    raw = '<tool_call>{"name": "calculate", "arguments": {"expression": "2400*0.15"}}</tool_call>'
    round_ = execute_round(raw)
    assert round_ is not None
    assert round_.fastpath == "That's 360."
    assert round_.tool_messages[0]["role"] == "assistant"
    assert round_.tool_messages[1]["role"] == "tool"
    assert "<tool_response>" in round_.tool_messages[1]["content"]


def test_execute_round_observation_tool_needs_composition(tmp_path, monkeypatch) -> None:
    (tmp_path / "n.md").write_text("The water cycle moves water around Earth.", "utf-8")
    monkeypatch.setattr(config, "NOTES_DIR", tmp_path)
    raw = '<tool_call>{"name": "search_notes", "arguments": {"query": "water cycle"}}</tool_call>'
    round_ = execute_round(raw)
    assert round_ is not None
    assert round_.fastpath is None  # model composes the spoken answer from the observation
    assert "water cycle" in round_.tool_messages[1]["content"].lower()


def test_execute_round_add_note_writes_file(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(config, "NOTES_DIR", tmp_path)
    raw = '<tool_call>{"name": "add_note", "arguments": {"content": "Exam is on Friday"}}</tool_call>'
    round_ = execute_round(raw)
    assert round_ is not None
    assert round_.fastpath == "Noted."
    assert (tmp_path / "memory.md").read_text("utf-8") == "- Exam is on Friday\n"
    # and the memory is searchable via search_notes (the remember -> recall circle)
    assert "Exam is on Friday" in search_notes("exam", notes_dir=tmp_path)


def test_execute_round_get_datetime_fastpath() -> None:
    raw = '<tool_call>{"name": "get_datetime", "arguments": {}}</tool_call>'
    round_ = execute_round(raw)
    assert round_ is not None
    assert round_.fastpath.startswith("It's ")
    assert round_.fastpath.endswith(".")


def test_speakable_filters_tags_and_json() -> None:
    assert speakable("Hello there.") == "Hello there."
    assert speakable('<tool_call>{"name": "calculate"}</tool_call>') is None
    assert speakable('{"tool": "calculate"}') is None
    assert speakable('prefix <tool_call>{"x": 1}</tool_call> tail') == "prefix  tail"


def test_tool_call_block_strips_unterminated_tag() -> None:
    assert TOOL_CALL_BLOCK.sub("", "text <tool_call>{\"x\": 1") == "text "
