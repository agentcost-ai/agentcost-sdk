"""
AgentCost capability fingerprinting.

Records *what a call needed*, so the optimizer can tell whether a cheaper model
would still work. The model catalogue already knows which models support vision
or tool calling; without this it has nothing to compare that against, treats the
requirement as unknown, and — because unknown reads as "not required" — happily
proposes a text-only model for a workload that sends images.

Only booleans and counts are recorded. No prompt text, no tool definitions, no
image data: the fingerprint has to be safe to send from a process that is
deliberately not sending any of that.
"""

from typing import Any, Dict, Optional

# Reserved metadata key. Namespaced so a caller's own metadata can never
# collide with it, and so the server can distinguish a fingerprint the SDK
# produced from keys a user happened to name "tools".
CAPABILITY_KEY = "_ac_caps"

_VISION_PART_TYPES = {"image_url", "input_image", "image"}

# Keys and attributes that carry binary media in a content part: `source` on
# Anthropic blocks, `inline_data`/`file_data` (and camelCase variants) on
# Gemini parts.
_MEDIA_KEYS = ("source", "inline_data", "inlineData", "file_data", "fileData")
_MEDIA_ATTRS = ("inline_data", "file_data")


def _part_is_media(part: Any) -> bool:
    """True if one content part carries an image or other binary media."""
    if isinstance(part, (str, bytes)) or part is None:
        return False
    if isinstance(part, dict):
        if part.get("type") in _VISION_PART_TYPES:
            return True
        return any(key in part for key in _MEDIA_KEYS)
    if getattr(part, "type", None) in _VISION_PART_TYPES:
        return True
    if any(getattr(part, attr, None) is not None for attr in _MEDIA_ATTRS):
        return True
    # google-genai accepts PIL images directly in a contents list.
    return type(part).__module__.split(".", 1)[0] == "PIL"


def _messages_have_images(messages: Any) -> bool:
    """True if any message carries non-text content, in any provider's shape.

    OpenAI and Anthropic nest parts under `content`; Gemini nests them under
    `parts`, and also accepts bare parts (or PIL images) at the top level.
    """
    if not isinstance(messages, (list, tuple)):
        return False
    for message in messages:
        if _part_is_media(message):
            return True
        if isinstance(message, dict):
            content = message.get("content") or message.get("parts")
        else:
            content = getattr(message, "content", None) or getattr(message, "parts", None)
        if not isinstance(content, (list, tuple)):
            continue
        for part in content:
            if _part_is_media(part):
                return True
    return False


def _structured_output(kwargs: Dict[str, Any]) -> bool:
    """True if the call pins the response to a schema or JSON object."""
    response_format = kwargs.get("response_format")
    if isinstance(response_format, dict):
        if response_format.get("type") in {"json_object", "json_schema"}:
            return True
    elif response_format is not None:
        # The Gemini and newer OpenAI SDKs accept a type/class here.
        return True
    return bool(kwargs.get("response_schema") or kwargs.get("response_mime_type"))


def fingerprint(kwargs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Summarise the capabilities one request actually exercised.

    Returns None when nothing notable was used, so the common case adds no
    bytes to the event.
    """
    try:
        tools = kwargs.get("tools") or kwargs.get("functions")
        tool_count = len(tools) if isinstance(tools, (list, tuple)) else 0

        has_images = _messages_have_images(
            kwargs.get("messages") or kwargs.get("input") or kwargs.get("contents")
        )

        caps: Dict[str, Any] = {}
        if has_images:
            caps["vision"] = True
        if tool_count or kwargs.get("tool_choice") or kwargs.get("function_call"):
            caps["tools"] = True
            if tool_count:
                caps["tool_count"] = tool_count
        if _structured_output(kwargs):
            caps["structured_output"] = True

        return caps or None
    except Exception:
        # Runs inside the interceptors' guarded paths. A capability hint is
        # never worth risking the caller's call or losing the event over.
        return None


def merge_config(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten a Gemini-style nested ``config`` into the top-level kwargs.

    google-genai carries tools and response schema inside ``config`` rather
    than as siblings of ``contents``, so inspecting kwargs alone would always
    report no capabilities for Gemini.
    """
    config = kwargs.get("config")
    if config is None:
        return kwargs

    merged = dict(kwargs)
    if isinstance(config, dict):
        merged.update(config)
    else:
        for attribute in (
            "tools",
            "tool_config",
            "response_schema",
            "response_mime_type",
        ):
            value = getattr(config, attribute, None)
            if value is not None:
                merged[attribute] = value
    return merged
