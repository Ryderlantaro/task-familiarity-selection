# Normalize raw instruction data for selection and LoRA training.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


CANONICAL_KEYS = ("instruction", "context", "response", "source_tag", "metadata")


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _format_history(role: str, content: str) -> str:
    label = "System" if role == "system" else "User" if role == "user" else "Assistant"
    return f"{label}: {content}"


def _role_name(msg: dict[str, Any]) -> str:
    role = _as_text(msg.get("role") or msg.get("from")).lower()
    if role in {"system"}:
        return "system"
    if role in {"user", "human", "prompter", "instruction"}:
        return "user"
    if role in {"assistant", "gpt", "bot", "model", "response", "output"}:
        return "assistant"
    return role or "unknown"


def _extract_messages(record: dict[str, Any]) -> tuple[str, str, str, str]:
    messages = record.get("messages") or record.get("conversations") or []
    turns: list[tuple[str, str]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = _role_name(msg)
        content = _as_text(msg.get("content") or msg.get("value") or msg.get("text"))
        if not content:
            continue
        if role not in {"system", "user", "assistant"}:
            role = "user" if not any(r == "user" for r, _ in turns) else "assistant"
        turns.append((role, content))

    if not turns:
        return "", "", "", "empty"

    last_assistant = None
    for i in range(len(turns) - 1, -1, -1):
        if turns[i][0] == "assistant":
            last_assistant = i
            break
    if last_assistant is None:
        context = "\n\n".join(_format_history(r, c) for r, c in turns[:-1]).strip()
        return turns[-1][1], context, "", "last_message_no_assistant"

    last_user = None
    for i in range(last_assistant - 1, -1, -1):
        if turns[i][0] == "user":
            last_user = i
            break

    if last_user is None:
        context = "\n\n".join(_format_history(r, c) for r, c in turns[:last_assistant]).strip()
        return "", context, turns[last_assistant][1], "last_assistant_no_user"

    context_turns = turns[:last_user]
    context = "\n\n".join(_format_history(r, c) for r, c in context_turns).strip()
    instruction = turns[last_user][1]
    response = turns[last_assistant][1]
    return instruction, context, response, "last_turn"


def normalize_record(record: dict[str, Any], raw_index: int | None = None) -> dict[str, Any]:
    # Normalize one raw dataset row to the project canonical schema.
    if not isinstance(record, dict):
        record = {"text": _as_text(record)}

    if "messages" in record or "conversations" in record:
        instruction, context, response, turn_mode = _extract_messages(record)
        raw_format = "messages" if "messages" in record else "conversations"
    else:
        turn_mode = "single_turn"
        raw_format = "alpaca_like"
        instruction = _as_text(
            record.get("instruction")
            or record.get("prompt")
            or record.get("question")
            or record.get("query")
            or record.get("input")
            or record.get("text")
        )
        context = _as_text(record.get("context") or record.get("input") or "")
        response = _as_text(
            record.get("response")
            or record.get("output")
            or record.get("answer")
            or record.get("completion")
            or record.get("target")
        )

    # Avoid duplicating Alpaca-style input used as a fallback instruction.
    if record.get("instruction") and record.get("input") and not record.get("context"):
        context = _as_text(record.get("input"))

    source_tag = _as_text(
        record.get("source_tag")
        or record.get("source")
        or record.get("dataset")
        or record.get("data_source")
        or "unknown"
    )

    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    metadata = dict(metadata)
    if raw_index is not None:
        metadata.setdefault("raw_index", raw_index)
    metadata.setdefault("raw_format", raw_format)
    metadata.setdefault("turn_mode", turn_mode)
    for key in ("id", "domain", "category", "task", "source", "dataset"):
        if key in record and key not in metadata:
            metadata[key] = record[key]

    out = {
        "instruction": instruction,
        "context": context,
        "response": response,
        "source_tag": source_tag,
        "metadata": metadata,
    }
    if "messages" in record:
        out["_raw_messages"] = record.get("messages")
    elif "conversations" in record:
        out["_raw_messages"] = record.get("conversations")
    return out


def normalize_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [normalize_record(row, raw_index=i) for i, row in enumerate(records)]


def load_jsonl_raw(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_jsonl_normalized(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line:
                rows.append(normalize_record(json.loads(line), raw_index=i))
    return rows


def canonical_text(record: dict[str, Any], include_response: bool = True) -> str:
    # Text used for selection representation: instruction + response + context.
    item = normalize_record(record)
    parts = [item["instruction"]]
    if include_response and item.get("response"):
        parts.append(item["response"])
    if item.get("context"):
        parts.append(item["context"])
    return "\n\n".join(p for p in parts if p).strip()


def raw_messages_text(record: dict[str, Any]) -> str:
    # Raw user/assistant/system contents in original order when available.
    if isinstance(record, dict) and ("messages" in record or "conversations" in record or "_raw_messages" in record):
        messages = record.get("messages") or record.get("conversations") or record.get("_raw_messages") or []
        parts = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            content = _as_text(msg.get("content") or msg.get("value") or msg.get("text"))
            if content:
                parts.append(content)
        if parts:
            return "\n\n".join(parts).strip()
    item = normalize_record(record)
    parts = [item["instruction"]]
    if item.get("context"):
        parts.append(item["context"])
    if item.get("response"):
        parts.append(item["response"])
    return "\n\n".join(p for p in parts if p).strip()


def answer_focused_text(record: dict[str, Any]) -> str:
    # Instruction + response only, excluding earlier conversation context.
    item = normalize_record(record)
    parts = [item["instruction"]]
    if item.get("response"):
        parts.append(item["response"])
    return "\n\n".join(p for p in parts if p).strip()


def assistant_response_text(record: dict[str, Any]) -> str:
    # Return only the final assistant response for response-side scoring.
    return normalize_record(record).get("response", "").strip()


def assistant_response_token_lengths(
    pool_path: str | Path,
    row_indices: list[int] | tuple[int, ...] | Any,
    tokenizer: Any,
) -> Any:
    # Count assistant-response tokens in the supplied pool-row order.
    import numpy as np

    indices = np.asarray(row_indices, dtype=np.int64)
    wanted = set(indices.tolist())
    rows: dict[int, dict[str, Any]] = {}
    with Path(pool_path).open(encoding="utf-8") as source:
        for row_index, line in enumerate(source):
            if row_index in wanted:
                rows[row_index] = json.loads(line)
    if len(rows) != len(indices):
        raise ValueError("some requested row indices are outside the candidate pool")
    return np.asarray(
        [
            len(tokenizer(assistant_response_text(rows[int(index)]), add_special_tokens=False)["input_ids"])
            for index in indices
        ],
        dtype=np.int32,
    )


def chat_template_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    # Return normalized messages for a model-native chat template.
    raw_messages = (
        record.get("messages")
        or record.get("conversations")
        or record.get("_raw_messages")
        or []
    )
    messages: list[dict[str, str]] = []
    for message in raw_messages:
        if not isinstance(message, dict):
            continue
        role = _role_name(message)
        content = _as_text(message.get("content") or message.get("value") or message.get("text"))
        if content and role in {"system", "user", "assistant"}:
            messages.append({"role": role, "content": content})
    if messages:
        return messages

    item = normalize_record(record)
    if item.get("context"):
        messages.append({"role": "system", "content": item["context"]})
    if item.get("instruction"):
        messages.append({"role": "user", "content": item["instruction"]})
    if item.get("response"):
        messages.append({"role": "assistant", "content": item["response"]})
    return messages


def selection_text(record: dict[str, Any], view: str = "canonical", tokenizer=None) -> str:
    if view == "canonical":
        return canonical_text(record)
    if view == "question_side":
        return normalize_record(record)["instruction"]
    if view == "raw_messages":
        return raw_messages_text(record)
    if view == "answer_focused":
        return answer_focused_text(record)
    if view == "assistant_response":
        text = assistant_response_text(record)
        if not text:
            raise ValueError("assistant_response view requires a non-empty assistant response")
        return text
    raise ValueError(f"unsupported selection text view: {view}")


def instruction_text(record: dict[str, Any]) -> str:
    item = normalize_record(record)
    parts = [item["instruction"]]
    if item.get("context"):
        parts.append(item["context"])
    return "\n\n".join(p for p in parts if p).strip()


def text_parts(record: dict[str, Any]) -> tuple[str, str]:
    item = normalize_record(record)
    prompt = instruction_text(item)
    return prompt, item.get("response", "")
