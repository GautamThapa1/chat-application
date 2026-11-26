"""Canonical Google GenAI function declarations and schema conversion."""

from __future__ import annotations

from typing import Any

from backend.service.gemini_chat import FUNCTION_DECLARATIONS

DECLARATIONS_BY_NAME = {
    declaration.name: declaration for declaration in FUNCTION_DECLARATIONS
}


def _schema_type(value: Any) -> str:
    if hasattr(value, "value"):
        value = value.value
    value = str(value or "OBJECT").lower()
    return {
        "string": "string",
        "integer": "integer",
        "number": "number",
        "boolean": "boolean",
        "object": "object",
        "array": "array",
    }.get(value, value)


def google_schema_to_json_schema(schema: Any) -> dict[str, Any]:
    """Convert google.genai.types.Schema to the JSON Schema used by tools."""
    result: dict[str, Any] = {"type": _schema_type(getattr(schema, "type", None))}

    description = getattr(schema, "description", None)
    if description:
        result["description"] = description

    properties = getattr(schema, "properties", None) or {}
    if properties:
        result["properties"] = {
            name: google_schema_to_json_schema(property_schema)
            for name, property_schema in properties.items()
        }

    required = getattr(schema, "required", None)
    if required:
        result["required"] = list(required)

    minimum = getattr(schema, "minimum", None)
    maximum = getattr(schema, "maximum", None)
    min_length = getattr(schema, "min_length", None)
    max_length = getattr(schema, "max_length", None)
    if minimum is not None:
        result["minimum"] = minimum
    if maximum is not None:
        result["maximum"] = maximum
    if min_length is not None:
        result["minLength"] = min_length
    if max_length is not None:
        result["maxLength"] = max_length

    enum = getattr(schema, "enum", None)
    if enum:
        result["enum"] = list(enum)

    items = getattr(schema, "items", None)
    if items is not None:
        result["items"] = google_schema_to_json_schema(items)

    return result


def tool_args_schema(tool_name: str) -> dict[str, Any]:
    """Return the JSON Schema input contract for one declared Gemini tool."""
    declaration = DECLARATIONS_BY_NAME.get(tool_name)
    if declaration is None:
        raise KeyError(f"No Gemini function declaration exists for {tool_name!r}.")

    parameters = declaration.parameters
    if parameters is None:
        return {"type": "object", "properties": {}}
    return google_schema_to_json_schema(parameters)


__all__ = [
    "DECLARATIONS_BY_NAME",
    "FUNCTION_DECLARATIONS",
    "google_schema_to_json_schema",
    "tool_args_schema",
]
