"""Snap's local tool layer — the entire agentic surface, kept deliberately small.

Design constraints:
- Tools are deterministic and local (no network). Demo reliability > feature count.
- The LLM emits ONE strict JSON decision per turn; the router executes it locally.
- Tool results are spoken from a template WITHOUT a second LLM call (latency trick).
"""

import ast
import json
import operator
from pathlib import Path

from snap import config

SYSTEM_PROMPT = """You are Snap, an offline voice assistant. Reply in the user's language (English or Hindi).
For normal conversation, answer with PLAIN TEXT only — never wrap it in JSON.
Only use JSON when you need a tool, as one single object and nothing else:
- calculate: {"tool": "calculate", "args": {"expression": "<arithmetic, e.g. 2400*0.15>"}}
- convert: {"tool": "convert", "args": {"value": <number>, "from_unit": "km|mi|kg|lb|c|f", "to_unit": "..."}}
- search_notes: {"tool": "search_notes", "args": {"query": "<topic>"}} — searches the user's personal notes
Use a tool for math, unit conversions, or questions about the user's notes.
Keep replies short — this is a voice assistant.
/no_think"""


# --- individual tools -------------------------------------------------------

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
    return f"From your note '{path.stem}': {best[:400]}"


# --- router -----------------------------------------------------------------

_TOOLS = {
    "calculate": lambda a: f"That's {calculate(a['expression']):g}.",
    "convert": lambda a: f"That's {convert(a['value'], a['from_unit'], a['to_unit']):g} {a['to_unit']}.",
    "search_notes": lambda a: search_notes(a["query"]),
}


def route(llm_output: str) -> str | None:
    """Parse the LLM's JSON decision. Returns the final spoken text, or None if the
    output was plain conversational JSON ({'say': ...}). Raises nothing: a malformed
    response is treated as a plain reply so the demo never dies mid-sentence."""
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
            return _TOOLS[decision["tool"]](decision.get("args", {}))
        except Exception as exc:  # noqa: BLE001 — spoken error beats a crash on camera
            return f"Sorry, that tool failed: {exc}"
    return text
