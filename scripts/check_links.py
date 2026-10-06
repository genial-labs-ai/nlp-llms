"""Check that every internal link in the rendered site points at a real file.

Reads docs/**/*.html after `quarto render`. External links are not fetched.

Also checks that no collapsed callout is left as Quarto's Bootstrap toggle, a
header that the keyboard cannot reach: filters/disclosure.lua writes them as
<details>. A Quarto upgrade that changes how callouts are parsed would fail here.

Run:  uv run --group site python scripts/check_links.py
"""

from __future__ import annotations

import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

DOCS = Path(__file__).resolve().parent.parent / "docs"


class LinkCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []
        self.toggles = 0

    def handle_starttag(self, tag, attrs):
        for name, value in attrs:
            if name in ("href", "src") and value:
                self.links.append(value)
        found = dict(attrs)
        if "data-bs-toggle" in found and "callout-header" in (found.get("class") or "").split():
            self.toggles += 1


def is_internal(link: str) -> bool:
    parts = urlsplit(link)
    return not parts.scheme and not parts.netloc and bool(parts.path)


def main() -> int:
    if not DOCS.is_dir():
        print("docs/ not found: run `quarto render` first")
        return 2
    pages = sorted(DOCS.rglob("*.html"))
    broken = []
    toggles = []
    for page in pages:
        collector = LinkCollector()
        collector.feed(page.read_text(encoding="utf-8"))
        if collector.toggles:
            toggles.append(f"{page.relative_to(DOCS)}: {collector.toggles}")
        for link in collector.links:
            if not is_internal(link):
                continue
            path = unquote(urlsplit(link).path)
            target = (DOCS / path.lstrip("/")) if path.startswith("/") else (page.parent / path)
            if target.is_dir():
                target = target / "index.html"
            if not target.exists():
                broken.append(f"{page.relative_to(DOCS)}: {link}")
    for line in broken:
        print(f"BROKEN  {line}")
    for line in toggles:
        print(
            f"TOGGLE  {line} collapsed callout(s) not written as <details> (filters/disclosure.lua)"
        )
    print(
        f"{len(pages)} pages checked, {len(broken)} broken internal links,"
        f" {len(toggles)} pages with callout toggles"
    )
    return 1 if broken or toggles else 0


if __name__ == "__main__":
    sys.exit(main())
