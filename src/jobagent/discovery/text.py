"""HTML-to-text conversion using only the standard library."""

from __future__ import annotations

import re
from html import unescape
from html.parser import HTMLParser

_SKIP_TAGS = frozenset({"script", "style", "noscript"})
_BLOCK_TAGS = frozenset(
    {
        "p",
        "div",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "tr",
        "table",
        "section",
        "article",
        "ul",
        "ol",
    }
)


class _HTMLToText(HTMLParser):
    """Collect visible text, inserting breaks for paragraphs, lists, and <br>."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag == "br":
            self._parts.append("\n")
        elif tag == "li":
            self._parts.append("\n- ")
        elif tag in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
            return
        if self._skip_depth:
            return
        if tag in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        self._parts.append(data)

    def text(self) -> str:
        joined = "".join(self._parts)
        lines: list[str] = []
        for raw_line in joined.splitlines():
            collapsed = re.sub(r"\s+", " ", raw_line).strip()
            if collapsed:
                lines.append(collapsed)
        return "\n".join(lines)


def clean_html(raw: str) -> str:
    """Turn HTML (including Greenhouse's escaped HTML) into readable plain text."""
    parser = _HTMLToText()
    parser.feed(unescape(raw))
    parser.close()
    return parser.text()
