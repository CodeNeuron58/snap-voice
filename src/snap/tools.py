"""Snap's tool layer — schema-driven tools wired Hermes-style, bounded like OpenClaw.

Patterns adopted from reference agents (Hermes function-calling; OpenClaw's ReAct loop):
- Tools are declared once as a registry with JSON schemas; the system prompt renders
  them inside <tools> tags.
- The model emits <tool_call>{"name", "arguments"}</tool_call>; observations go back
  as <tool_response> so the model can compose the spoken answer (or call again).
- Tool errors are fed back to the model too — it recovers instead of the demo dying.
- The loop is hard-capped (config.MAX_TOOL_HOPS) — no runaway ReAct cycles.

Latency guard: deterministic tools carry a speak template, so calculate / convert /
add_note / get_datetime answers never pay a second LLM pass. Only search_notes
(where composition genuinely helps) takes the extra pass.
"""

import ast
import json
import operator
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from snap import config

# --- deterministic tools (pure functions) -------------------------------------

_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg,
    ast.Mod: operator.mod,
}


def calculate(expression: str) -> float:
    """Safe arithmetic via AST whitelist — no eval of arbitrary code."""

    def _ev(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](_ev(node.left), _ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](_ev(node.operand))
        raise ValueError(f"unsupported expression: {ast.dump(node)}")

    return float(_ev(ast.parse(expression, mode="eval").body))


_UNITS = {
    ("km", "mi"): 0.621371, ("mi", "km"): 1.609344,
    ("kg", "lb"): 2.20462, ("lb", "kg"): 0.453592,
    ("c", "f"): None, ("f", "c"): None,
}


def convert(value: float, from_unit: str, to_unit: str) -> float:
    key = (from_unit.lower(), to_unit.lower())
    if key not in _UNITS:
        raise ValueError(f"unsupported unit pair: {key}")
    if key == ("c", "f"):
        return value * 9 / 5 + 32
    if key == ("f", "c"):
        return (value - 32) * 5 / 9
    return value * _UNITS[key]


def search_notes(query: str, notes_dir: Path | None = None) -> str:
    """Naive keyword scoring over markdown notes. Swap for an on-device embedding
    search later only if demo quality demands it."""
    notes_dir = notes_dir or config.NOTES_DIR
    terms = [t for t in query.lower().split() if len(t) > 2]
    scored: list[tuple[int, Path, str]] = []
    for path in notes_dir.glob("*.md"):
        text = path.read_text(encoding="utf-8")
        score = sum(text.lower().count(t) for t in terms)
        if score:
            scored.append((score, path, text))
    if not scored:
        return f"No notes matched '{query}'."
    scored.sort(reverse=True)
    _, path, text = scored[0]
    # Speak the paragraph containing the best-scoring term, not the whole file.
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    best = max(paragraphs, key=lambda p: sum(p.lower().count(t) for t in terms))
    best = re.sub(r"^#+\s*", "", best)  # markdown headings would be spoken as "hashtag"
    return f"From your note '{path.stem}': {best[:400]}"


def add_note(content: str) -> str:
    """Append a memory to the notes inbox — 'remember that …' becomes searchable."""
    content = content.strip()
    if not content:
        raise ValueError("content is required")
    config.NOTES_DIR.mkdir(parents=True, exist_ok=True)
    with (config.NOTES_DIR / "memory.md").open("a", encoding="utf-8") as f:
        f.write(f"- {content}\n")
    return f"Saved to your notes: {content}"


def get_datetime() -> str:
    import datetime as _dt

    return _dt.datetime.now().strftime("%H:%M, %A %d %B %Y")


# --- tool registry --------------------------------------------------------------

@dataclass(frozen=True)
class Tool:
    """One tool: JSON schema (for the prompt) + handler (observation) + optional
    speak template (fast-path spoken answer that skips the second LLM pass)."""

    name: str
    description: str
    parameters: dict
    handler: Callable[[dict], str]
    speak: Callable[[str], str] | None = None  # observation -> spoken answer


def _openai_schema(tool: Tool) -> dict:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }


TOOLS: dict[str, Tool] = {
    t.name: t
    for t in (
        Tool(
            name="calculate",
            description="Evaluate an arithmetic expression. Use for ANY math.",
            parameters={
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
            handler=lambda a: repr(calculate(a["expression"])),
            speak=lambda obs: f"That's {float(obs):g}.",
        ),
        Tool(
            name="convert",
            description="Convert between km, mi, kg, lb, celsius and fahrenheit.",
            parameters={
                "type": "object",
                "properties": {
                    "value": {"type": "number"},
                    "from_unit": {"type": "string", "enum": ["km", "mi", "kg", "lb", "c", "f"]},
                    "to_unit": {"type": "string", "enum": ["km", "mi", "kg", "lb", "c", "f"]},
                },
                "required": ["value", "from_unit", "to_unit"],
            },
            handler=lambda a: repr(convert(a["value"], a["from_unit"], a["to_unit"])),
            speak=lambda obs, *a: obs,  # replaced below — needs from_unit/to_unit
        ),
        Tool(
            name="search_notes",
            description="Search the user's personal notes. Use for 'in my notes…' questions.",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
            handler=lambda a: search_notes(a["query"]),
            speak=None,  # composed by the model from the observation
        ),
        Tool(
            name="add_note",
            description="Save a memory to the user's notes. Use for 'remember that…'.",
            parameters={
                "type": "object",
                "properties": {"content": {"type": "string"}},
                "required": ["content"],
            },
            handler=lambda a: add_note(a["content"]),
            speak=lambda obs: "Noted.",
        ),
        Tool(
            name="get_datetime",
            description="Get the current local date and time. Use for 'what time/date'.",
            parameters={"type": "object", "properties": {}},
            handler=lambda a: get_datetime(),
            speak=lambda obs: f"It's {obs}.",
        ),
    )
}


def _convert_speak(args: dict, observation: str) -> str:
    return f"That's {float(observation):g} {args['to_unit']}."


# convert's template needs the args, so it is special-cased in execute_round.

def schemas_block() -> str:
    """Render the <tools> block (Hermes-style) embedded in the system prompt."""
    rendered = ",\n".join(
        json.dumps(_openai_schema(t)["function"], ensure_ascii=False) for t in TOOLS.values()
    )
    return f"<tools>\n[{rendered}]\n</tools>"


def openai_schemas() -> list[dict]:
    """OpenAI-style tools list (accepted by llama-cpp-python's tools= parameter)."""
    return [_openai_schema(t) for t in TOOLS.values()]


# --- Hermes-style protocol: <tool_call> out, <tool_response> back ----------------

_TOOL_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
# A tag left unterminated (stream cut by token budget) must still never be spoken.
TOOL_CALL_BLOCK = re.compile(r"<tool_call>.*?(?:</tool_call>|\Z)", re.S | re.I)


def parse_tool_calls(text: str) -> list[tuple[str, dict]]:
    """Extract (name, args) pairs from <tool_call> blocks. Tolerant: malformed or
    unterminated blocks are skipped (never raise — the demo must not die on model
    drift). Falls back to the legacy {'tool': …, 'args': …} JSON if no tags exist."""
    calls: list[tuple[str, dict]] = []
    for match in _TOOL_CALL.finditer(text):
        try:
            obj = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        name = obj.get("name") or obj.get("tool")
        args = obj.get("arguments")
        if not isinstance(args, dict):
            args = obj.get("args") if isinstance(obj.get("args"), dict) else {}
        if name:
            calls.append((str(name), args))
    if calls:
        return calls
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            obj = json.loads(stripped)
            name = obj.get("tool") or obj.get("name")
            args = obj.get("args") if isinstance(obj.get("args"), dict) else obj.get("arguments")
            if name and isinstance(args, dict):
                calls.append((str(name), args))
        except json.JSONDecodeError:
            pass
    return calls


def dispatch(name: str, args: dict) -> str:
    """Execute one tool and return the OBSERVATION string. Never raises: failures
    come back as a Hermes-style error so the model can correct and retry."""
    tool = TOOLS.get(name)
    if tool is None:
        return (
            f"There was an error when executing the function: {name} is not an "
            "available tool. Available tools: " + ", ".join(TOOLS) + "."
        )
    try:
        return tool.handler(args)
    except Exception as exc:  # noqa: BLE001 — the model sees the error, the user never does
        return (
            f"There was an error when executing the function: {name}\n{exc}\n"
            "Please correct the arguments and try again."
        )


@dataclass
class ToolRound:
    """One round of tool execution on a model response."""

    tool_messages: list[dict]  # assistant call + tool responses, to feed back
    fastpath: str | None  # spoken answer if every tool had a template (no 2nd pass)
    raw_calls: list[tuple[str, dict]]


def execute_round(raw: str) -> ToolRound | None:
    """If the model output contains tool calls, run them and package the feedback.

    `fastpath` is set when the last tool has a speak template — the caller speaks it
    directly (zero extra LLM latency). Otherwise the caller feeds `tool_messages`
    back for another pass (composed answer). Returns None for plain-text output.
    """
    calls = parse_tool_calls(raw)
    if not calls:
        return None
    tool_messages: list[dict] = [{"role": "assistant", "content": raw.strip()}]
    last_result = ""
    last_tool: Tool | None = None
    last_args: dict = {}
    for name, args in calls:
        observation = dispatch(name, args)
        tool_messages.append(
            {"role": "tool", "content": f"<tool_response>\n{observation}\n</tool_response>"}
        )
        last_result, last_tool, last_args = observation, TOOLS.get(name), args
    fastpath = None
    if last_tool is not None and last_tool.speak is not None:
        if last_tool.name == "convert":
            fastpath = _convert_speak(last_args, last_result)
        else:
            fastpath = last_tool.speak(last_result)
    return ToolRound(tool_messages=tool_messages, fastpath=fastpath, raw_calls=calls)


def speakable(sentence: str) -> str | None:
    """Cleaned speakable text for a streamed sentence — None for tool JSON/tags.

    Shared by both paths (pipecat + legacy) so what counts as speakable can't drift.
    """
    cleaned = TOOL_CALL_BLOCK.sub("", sentence).strip()
    if not cleaned or cleaned.lstrip().startswith("{"):
        return None  # raw tool JSON never reaches speech; the router speaks the result
    return cleaned


SYSTEM_PROMPT = f"""You are Snap, an offline voice assistant. If the user writes in Hindi or Hinglish (even one Hindi word), reply in Hindi. Otherwise reply in English.

You have access to the following tools:
{schemas_block()}

Tool rules:
- To use a tool, reply with ONLY a tool call and nothing else:
  <tool_call>{{"name": "<tool_name>", "arguments": {{...}}}}</tool_call>
- ALWAYS use the calculate tool for any arithmetic and the convert tool for any unit
  conversion — never compute those yourself.
- After a tool_response, answer the user in PLAIN TEXT only — short, spoken-style.
  Never mention tags, JSON, tools or notes files; just answer naturally.
- For normal conversation, skip tools and answer in plain text.
Keep replies short — this is a voice assistant. Never use emojis or emoticons."""


# --- deterministic pre-router -------------------------------------------------
# Qwen3 with thinking disabled computes arithmetic badly and its tool adherence
# costs a prefill — so math and unit conversions never reach the LLM at all.
# A regex pre-router catches the deterministic cases, computes locally, and the
# LLM handles everything else. Zero tokens, zero hallucination, ~0ms answers.

_NUM = r"[\d,]+(?:\.\d+)?"
_UNIT = r"km|mi|miles?|kg|kilos?|pounds?|lbs?|celsius|fahrenheit"


def _digits(s: str) -> str:
    return s.replace(",", "")


def _h_percent(m: "re.Match[str]") -> str:
    a, b = _digits(m.group(1)), _digits(m.group(2))
    return f"That's {calculate(f'{a}*{b}/100'):g}."


def _h_sqrt(m: "re.Match[str]") -> str:
    n = _digits(m.group(1))
    return f"The square root of {n} is {calculate(n) ** 0.5:g}."


def _h_convert(m: "re.Match[str]") -> str:
    value, src, dst = float(_digits(m.group(1))), _canon(m.group(2)), _canon(m.group(3))
    return f"That's {convert(value, src, dst):g} {dst}."


def _h_binary(m: "re.Match[str]") -> str:
    a, b = _digits(m.group(1)), _digits(m.group(3))
    return f"That's {calculate(f'{a} {_op(m.group(2))} {b}'):g}."


_PATTERNS = [
    # "15% of 2400" / "15 percent of 2400"
    (re.compile(rf"({_NUM})\s*(?:%|percent)\s*of\s*({_NUM})", re.I), _h_percent),
    # "square root of 1764"
    (re.compile(rf"square root of\s*({_NUM})", re.I), _h_sqrt),
    # "convert 5 km to miles" / "12 kg kitne pound hote hain" / "3 km mein kitne miles"
    (
        re.compile(
            rf"(?:convert\s*)?({_NUM})\s*({_UNIT})\s*(?:to|in|into|kitne|hote|hote hain|mein)\s*(?:kitne\s+)?({_UNIT})",
            re.I,
        ),
        _h_convert,
    ),
    # binary arithmetic: "2400 * 12" / "456 times 89" / "12 plus 3" / "9 divided by 2"
    (
        re.compile(rf"({_NUM})\s*(\+|plus|-|minus|\*|x|×|times|into|/|divided by)\s*({_NUM})", re.I),
        _h_binary,
    ),
]

_OP_WORDS = {
    "+": "+", "plus": "+", "-": "-", "minus": "-",
    "*": "*", "x": "*", "×": "*", "times": "*", "into": "*",
    "/": "/", "divided by": "/",
}
_UNIT_WORDS = {"miles": "mi", "mile": "mi", "mi": "mi", "kilos": "kg", "kilo": "kg",
               "kg": "kg", "pounds": "lb", "pound": "lb", "lbs": "lb", "lb": "lb",
               "celsius": "c", "c": "c", "fahrenheit": "f", "f": "f"}


def _canon(unit: str) -> str:
    return _UNIT_WORDS.get(unit.lower().strip(), unit.lower())


def _op(word: str) -> str:
    return _OP_WORDS[word.lower()]


def pre_route(user_text: str) -> str | None:
    """Deterministic answers for arithmetic/convert phrasings; None if not matched."""
    if not config.PRE_ROUTE_ENABLED:
        return None
    for pattern, handler in _PATTERNS:
        m = pattern.search(user_text)
        if m:
            try:
                return handler(m)
            except Exception:  # noqa: BLE001 — fall through to the LLM
                continue
    return None


# --- legacy router (compat) ---------------------------------------------------

def route(llm_output: str) -> str | None:
    """Parse a legacy JSON decision ({'tool': …} / {'say': …}). Kept for the rare
    model drift to plain-JSON output; the primary path is parse_tool_calls."""
    text = llm_output.strip()
    try:
        start, end = text.index("{"), text.rindex("}") + 1
        decision = json.loads(text[start:end])
    except (ValueError, json.JSONDecodeError):
        return text or "…"
    if "say" in decision:
        return decision["say"]
    if "tool" in decision:
        try:
            return _TOOLS_LEGACY[decision["tool"]](decision.get("args", {}))
        except Exception as exc:  # noqa: BLE001 — spoken error beats a crash on camera
            return f"Sorry, that tool failed: {exc}"
    return text


_TOOLS_LEGACY = {
    "calculate": lambda a: f"That's {calculate(a['expression']):g}.",
    "convert": lambda a: f"That's {convert(a['value'], a['from_unit'], a['to_unit']):g} {a['to_unit']}.",
    "search_notes": lambda a: search_notes(a["query"]),
}
