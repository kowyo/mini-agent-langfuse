"""Session ID mapping between mini-agent and Langfuse."""

import os
from datetime import UTC, datetime

from langfuse import propagate_attributes

LANGFUSE_ENV_PUBLIC_KEY = "LANGFUSE_PUBLIC_KEY"
LANGFUSE_ENV_SECRET_KEY = "LANGFUSE_SECRET_KEY"


def is_langfuse_configured() -> bool:
    return bool(
        os.getenv(LANGFUSE_ENV_PUBLIC_KEY) and os.getenv(LANGFUSE_ENV_SECRET_KEY)
    )


def build_session_metadata(
    model: str | None = None,
    cwd: str | None = None,
    tags: list[str] | None = None,
) -> dict[str, str]:
    metadata: dict[str, str] = {"source": "mini-agent", "started_at": datetime.now(UTC).isoformat()}
    if model:
        metadata["model"] = model
    if cwd:
        metadata["cwd"] = cwd
    if tags:
        metadata["tags"] = ",".join(tags)
    return metadata


class LangfuseSessionScope:
    """Context manager that propagates session_id to all Langfuse observations."""

    def __init__(
        self,
        session_id: str,
        model: str | None = None,
        cwd: str | None = None,
        tags: list[str] | None = None,
    ) -> None:
        self.session_id = session_id
        self._propagator = propagate_attributes(
            session_id=session_id,
            metadata=build_session_metadata(model=model, cwd=cwd, tags=tags),
        )

    def __enter__(self) -> str:
        self._propagator.__enter__()
        return self.session_id

    def __exit__(self, *exc: object) -> None:
        self._propagator.__exit__(*exc)
