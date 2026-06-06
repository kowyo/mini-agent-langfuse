"""Langfuse observation recording for conversation turns."""

from typing import Any

from langfuse import get_client


def _to_dict(obj: Any) -> Any:
    """Convert an object to a plain JSON-safe dict recursively."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json", exclude_none=True)
    if isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_dict(i) for i in obj]
    return obj


def record_turn_observation(
    *,
    trace_name: str,
    user_message: str,
    assistant_message: str | None = None,
    tool_calls: list[Any] | None = None,
    tool_results: list[Any] | None = None,
    model: str | None = None,
    **usage: int,
) -> None:
    client = get_client()
    input_data: dict[str, Any] = {"message": user_message}
    if tool_results:
        input_data["tool_results"] = _to_dict(tool_results)
    output_data: dict[str, Any] = {}
    if assistant_message:
        output_data["message"] = assistant_message
    if tool_calls:
        output_data["tool_calls"] = _to_dict(tool_calls)

    obs = client.start_observation(
        name=trace_name,
        as_type="generation",
        model=model or "unknown",
        input=input_data,
        output=output_data,
        usage_details=usage or None,
    )
    obs.end()


def flush() -> None:
    get_client().flush()
