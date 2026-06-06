"""Langfuse observation recording for conversation turns."""

from typing import Any

from langfuse import get_client


def record_turn_observation(
    *,
    trace_name: str,
    user_message: str,
    assistant_message: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    tool_results: list[dict[str, Any]] | None = None,
    model: str | None = None,
    **usage: int,
) -> None:
    client = get_client()
    input_data: dict[str, Any] = {"message": user_message}
    if tool_results:
        input_data["tool_results"] = tool_results
    output_data: dict[str, Any] = {}
    if assistant_message:
        output_data["message"] = assistant_message
    if tool_calls:
        output_data["tool_calls"] = tool_calls

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
