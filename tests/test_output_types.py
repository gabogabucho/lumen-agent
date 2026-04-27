"""Tests for structured output types module."""

import json
import unittest

from lumen.core.output_types import OutputType, StructuredOutput


class TestOutputType(unittest.TestCase):
    def test_all_types_exist(self):
        assert OutputType.TEXT.value == "text"
        assert OutputType.DOCUMENT.value == "document"
        assert OutputType.NOTIFICATION.value == "notification"
        assert OutputType.WEB.value == "web"
        assert OutputType.IMAGE.value == "image"
        assert OutputType.PLOT.value == "plot"


class TestStructuredOutputCreation(unittest.TestCase):
    def test_default_is_text(self):
        o = StructuredOutput(content="hello")
        assert o.type == OutputType.TEXT
        assert o.content == "hello"

    def test_auto_generates_id(self):
        o1 = StructuredOutput(content="a")
        o2 = StructuredOutput(content="b")
        assert o1.output_id != o2.output_id
        assert len(o1.output_id) == 12

    def test_auto_generates_timestamp(self):
        import time

        before = time.time()
        o = StructuredOutput(content="x")
        assert o.timestamp >= before

    def test_explicit_id_and_timestamp(self):
        o = StructuredOutput(output_id="custom123", timestamp=1000.0)
        assert o.output_id == "custom123"
        assert o.timestamp == 1000.0


class TestShorthandFactories(unittest.TestCase):
    def test_text(self):
        o = StructuredOutput.text("hello", session_id="s1")
        assert o.type == OutputType.TEXT
        assert o.content == "hello"
        assert o.session_id == "s1"

    def test_document(self):
        o = StructuredOutput.document("# Title\nBody", title="My Doc")
        assert o.type == OutputType.DOCUMENT
        assert o.metadata["title"] == "My Doc"

    def test_notification(self):
        o = StructuredOutput.notification("Update ready", level="success")
        assert o.type == OutputType.NOTIFICATION
        assert o.metadata["level"] == "success"

    def test_web(self):
        o = StructuredOutput.web("<h1>Hi</h1>")
        assert o.type == OutputType.WEB
        assert o.content == "<h1>Hi</h1>"

    def test_image(self):
        o = StructuredOutput.image(
            "https://example.com/img.png", alt="Photo", mime_type="image/png"
        )
        assert o.type == OutputType.IMAGE
        assert o.metadata["alt"] == "Photo"
        assert o.metadata["mime_type"] == "image/png"

    def test_plot(self):
        o = StructuredOutput.plot('{"data": [1,2,3]}', plot_type="bar")
        assert o.type == OutputType.PLOT
        assert o.metadata["plot_type"] == "bar"


class TestSerialization(unittest.TestCase):
    def test_to_dict(self):
        o = StructuredOutput.text("test", session_id="s1")
        d = o.to_dict()
        assert d["type"] == "text"
        assert d["content"] == "test"
        assert d["session_id"] == "s1"
        assert "output_id" in d
        assert "timestamp" in d

    def test_from_dict(self):
        d = {
            "type": "notification",
            "content": "hello",
            "session_id": "s1",
            "metadata": {"level": "warn"},
        }
        o = StructuredOutput.from_dict(d)
        assert o.type == OutputType.NOTIFICATION
        assert o.content == "hello"
        assert o.metadata["level"] == "warn"

    def test_from_dict_invalid_type_fallback(self):
        d = {"type": "invalid_type", "content": "hello"}
        o = StructuredOutput.from_dict(d)
        assert o.type == OutputType.TEXT

    def test_to_json_roundtrip(self):
        o = StructuredOutput.document("body", title="T", session_id="s1")
        json_str = o.to_json()
        restored = StructuredOutput.from_dict(json.loads(json_str))
        assert restored.type == OutputType.DOCUMENT
        assert restored.content == "body"
        assert restored.metadata["title"] == "T"

    def test_is_plain_text(self):
        assert StructuredOutput.text("hi").is_plain_text() is True
        assert StructuredOutput.web("<b>hi</b>").is_plain_text() is False
        assert StructuredOutput.image("url").is_plain_text() is False


if __name__ == "__main__":
    unittest.main()
