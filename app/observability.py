"""Secret-safe structured logging helper for the support agent."""

import json
import re
from typing import Any

_SECRET_PATTERNS = [
    (re.compile(r"\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}\\b"), "[REDACTED]"),
    (re.compile(r"(?i)(password|passwd|secret|api[_ -]?key|token)\\s*[:=]\\s*\\S+"), r"\\1=[REDACTED]"),
    (re.compile(r"(?<!\\d)(?:\\d[ -]?){10,}(?!\\d)"), "[REDACTED]"),
]

def _sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(v) for v in value]
    if isinstance(value, str):
        text = value
        for pattern, replacement in _SECRET_PATTERNS:
            text = pattern.sub(replacement, text)
        return text
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return "[REDACTED]" if isinstance(value, int) and value > 9 else value
    return value

def log_event(event: str, **fields: Any) -> None:
    """Print one JSON event after redacting common sensitive values."""
    payload = {"event": event, **_sanitize(fields)}
    print(json.dumps(payload, ensure_ascii=False))
