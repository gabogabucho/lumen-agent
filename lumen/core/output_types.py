"""Structured output types — typed responses beyond plain text.

Lumen can return different output types (text, document, notification,
web, image, plot) depending on the tool/action that generated the result.
This module provides the type system and serialization for structured outputs.

Backward compatible: if no type is specified, output defaults to "text".
"""

import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class OutputType(str, Enum):
    TEXT = "text"
    DOCUMENT = "document"
    NOTIFICATION = "notification"
    WEB = "web"
    IMAGE = "image"
    PLOT = "plot"


@dataclass
class StructuredOutput:
    """A typed output artifact from the agent.

    Attributes:
        type: The output type enum.
        content: The primary content (text, HTML, markdown, URL, etc).
        metadata: Additional type-specific metadata.
        output_id: Unique identifier for this output.
        session_id: Session that generated this output.
        timestamp: Unix timestamp of creation.
    """

    type: OutputType = OutputType.TEXT
    content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    output_id: str = ""
    session_id: str = ""
    timestamp: float = 0.0

    def __post_init__(self):
        if not self.output_id:
            self.output_id = uuid.uuid4().hex[:12]
        if not self.timestamp:
            self.timestamp = time.time()

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-friendly dict."""
        return {
            "output_id": self.output_id,
            "type": self.type.value,
            "content": self.content,
            "metadata": self.metadata,
            "session_id": self.session_id,
            "timestamp": self.timestamp,
        }

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StructuredOutput":
        """Deserialize from dict."""
        output_type = OutputType.TEXT
        if isinstance(data.get("type"), str):
            try:
                output_type = OutputType(data["type"])
            except ValueError:
                pass
        return cls(
            type=output_type,
            content=str(data.get("content", "")),
            metadata=data.get("metadata") if isinstance(data.get("metadata"), dict) else {},
            output_id=str(data.get("output_id", "")),
            session_id=str(data.get("session_id", "")),
            timestamp=float(data.get("timestamp", 0)),
        )

    def is_plain_text(self) -> bool:
        """Check if this output is plain text (no special rendering needed)."""
        return self.type == OutputType.TEXT

    @classmethod
    def text(cls, content: str, session_id: str = "", **metadata: Any) -> "StructuredOutput":
        """Shorthand for text output."""
        return cls(type=OutputType.TEXT, content=content, session_id=session_id, metadata=metadata)

    @classmethod
    def document(cls, content: str, session_id: str = "", title: str = "", **metadata: Any) -> "StructuredOutput":
        """Shorthand for document output (markdown/html)."""
        meta = {"title": title, **metadata}
        return cls(type=OutputType.DOCUMENT, content=content, session_id=session_id, metadata=meta)

    @classmethod
    def notification(cls, content: str, session_id: str = "", level: str = "info", **metadata: Any) -> "StructuredOutput":
        """Shorthand for notification output."""
        meta = {"level": level, **metadata}
        return cls(type=OutputType.NOTIFICATION, content=content, session_id=session_id, metadata=meta)

    @classmethod
    def web(cls, content: str, session_id: str = "", **metadata: Any) -> "StructuredOutput":
        """Shorthand for renderable web content (HTML)."""
        return cls(type=OutputType.WEB, content=content, session_id=session_id, metadata=metadata)

    @classmethod
    def image(cls, url_or_base64: str, session_id: str = "", alt: str = "", mime_type: str = "", **metadata: Any) -> "StructuredOutput":
        """Shorthand for image output."""
        meta = {"alt": alt, "mime_type": mime_type, **metadata}
        return cls(type=OutputType.IMAGE, content=url_or_base64, session_id=session_id, metadata=meta)

    @classmethod
    def plot(cls, data: str, session_id: str = "", plot_type: str = "", **metadata: Any) -> "StructuredOutput":
        """Shorthand for plot/chart data (matplotlib/plotly JSON)."""
        meta = {"plot_type": plot_type, **metadata}
        return cls(type=OutputType.PLOT, content=data, session_id=session_id, metadata=meta)
