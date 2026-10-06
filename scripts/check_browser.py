"""Desktop browser checks of the rendered site, in Chromium through Playwright.

Serves docs/ on a local port and opens it at the laptop and desktop sizes the site
is built for (AGENTS.md, "Target devices"), in the light and the dark theme:

- 1280 x 800, 1440 x 900 and 1920 x 1080;
- 1366 x 768 at 150% zoom (a 911 x 512 CSS-pixel viewport at device scale 1.5).

On every page it checks that the requested theme applied, that the page does not
scroll sideways, that the navbar has the expected entries, that no script error or
failed request is logged, that every math element was typeset by KaTeX, and that
every demo drew without an Observable error. On one lecture per size and theme it
checks the keyboard path through an answer: Tab reaches its <summary>, the focus
ring is visible, Enter opens it and Space closes it.

Phones and tablets are out of scope. Run after `quarto render`:

    uv run --group browser python scripts/check_browser.py [--all-sizes-all-pages]

First run on a machine: `uv run --group browser playwright install chromium`.
By default every page is checked at 1280 x 800 in both themes, and a set of
representative pages at the other sizes.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import sys
import threading
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

DOCS = Path(__file__).resolve().parent.parent / "docs"
SIZES = {
    "1280x800": {"viewport": {"width": 1280, "height": 800}, "device_scale_factor": 1},
    "1440x900": {"viewport": {"width": 1440, "height": 900}, "device_scale_factor": 1},
    "1920x1080": {"viewport": {"width": 1920, "height": 1080}, "device_scale_factor": 1},
    "1366x768@150%": {"viewport": {"width": 911, "height": 512}, "device_scale_factor": 1.5},
}
THEMES = ("light", "dark")
NAVBAR = ["Home", "Start here", "Schedule", "Days", "Notebooks", "References", "Teach"]
# Pages checked at every size; all pages are checked at the first size.
REPRESENTATIVE = [
    "index.html",
    "prepare.html",
    "schedule.html",
    "day-4.html",
    "lectures/01-text-as-data.html",
    "lectures/06-pretraining-huggingface.html",
    "lectures/11-calibration.html",
    "lectures/12-rlcd-jev.html",
]
KEYBOARD_PAGE = "lectures/01-text-as-data.html"
DEMO_TIMEOUT_MS = 20_000


def serve(root: Path) -> tuple[http.server.ThreadingHTTPServer, str]:
    handler = functools.partial(QuietHandler, directory=str(root))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}/"


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args) -> None:
        pass


def theme_script(theme: str) -> str:
    """Quarto reads the stored scheme on load; "alternate" is the dark theme."""
    value = "alternate" if theme == "dark" else "default"
    return f"try {{ localStorage.setItem('quarto-color-scheme', '{value}'); }} catch (e) {{}}"


def check_page(page: Page, url: str, theme: str) -> list[str]:
    problems: list[str] = []
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(f"script error: {e}"))
    page.on(
        "console",
        lambda m: errors.append(f"console error: {m.text}") if m.type == "error" else None,
    )
    page.on(
        "requestfailed",
        lambda r: errors.append(f"request failed: {r.url} ({r.failure})"),
    )
    page.goto(url, wait_until="load")

    dark = page.evaluate("document.body.classList.contains('quarto-dark')")
    if dark != (theme == "dark"):
        problems.append(f"the {theme} theme did not apply")

    if page.evaluate("document.documentElement.scrollWidth > window.innerWidth + 1"):
        width = page.evaluate("document.documentElement.scrollWidth")
        problems.append(f"scrolls sideways: page is {width}px wide")

    # The icon links on the right (GitHub) have no text.
    found = [t.strip() for t in page.locator(".navbar-nav .nav-link .menu-text").all_inner_texts()]
    found = [t for t in found if t]
    if found != NAVBAR:
        problems.append(f"navbar is {found}, expected {NAVBAR}")

    untypeset = page.evaluate(
        "[...document.querySelectorAll('span.math')].filter(e => !e.querySelector('.katex')).length"
    )
    if untypeset:
        problems.append(f"{untypeset} math element(s) not typeset")

    demos = page.locator(".demo")
    for i in range(demos.count()):
        demo = demos.nth(i)
        demo.scroll_into_view_if_needed()
        try:
            demo.locator("svg, figure, form, table").first.wait_for(timeout=DEMO_TIMEOUT_MS)
        except Exception:
            problems.append(f"demo {i + 1} drew nothing in {DEMO_TIMEOUT_MS // 1000} s")
        if demo.locator(".observablehq--error").count():
            text = demo.locator(".observablehq--error").first.inner_text()[:160]
            problems.append(f"demo {i + 1} has an Observable error: {text}")

    return problems + errors


def check_keyboard(page: Page, url: str) -> list[str]:
    """Tab to the first answer's <summary>, see the focus ring, open with Enter, close
    with Space."""
    page.goto(url, wait_until="load")
    summary = page.locator("details.callout-details > summary").first
    if not summary.count():
        return ["no <details> answer on the page"]
    # Start from the element just before the first answer, then Tab once.
    found = page.evaluate(
        """() => {
          const s = document.querySelector('details.callout-details > summary');
          s.scrollIntoView({block: 'center'});
          const focusables = [...document.querySelectorAll(
              'a[href], button, summary, input, select')]
            .filter(e => e.offsetParent !== null);
          const at = focusables.indexOf(s);
          if (at < 0) return 'hidden';
          if (at === 0) return 'first';
          focusables[at - 1].focus();
          return 'ok';
        }"""
    )
    if found == "hidden":
        return ["the first answer's <summary> is not visible"]
    if found == "first":
        return ["the first answer has no visible focusable element before it to Tab from"]
    page.keyboard.press("Tab")
    focused = page.evaluate(
        "document.activeElement === document.querySelector('details.callout-details > summary')"
    )
    if not focused:
        return ["Tab did not reach the first answer's <summary>"]
    problems = []
    outline = page.evaluate("getComputedStyle(document.activeElement).outlineStyle")
    if outline in ("none", ""):
        problems.append("no visible focus ring on the focused <summary>")
    details = page.locator("details.callout-details").first
    page.keyboard.press("Enter")
    if not details.evaluate("d => d.open"):
        problems.append("Enter did not open the answer")
    page.keyboard.press("Space")
    if details.evaluate("d => d.open"):
        problems.append("Space did not close the answer")
    return problems


def run(check, context, *args) -> list[str]:
    """A check's problems, on a new page of the context; an exception (a timeout, a
    crashed renderer, a script that threw) is one more, so one bad page is reported and
    the run goes on."""
    page = None
    try:
        page = context.new_page()
        return check(page, *args)
    except Exception as e:  # any failure here is a finding, not a crash
        first = (str(e).splitlines() or [""])[0]
        return [f"the check itself failed: {type(e).__name__}: {first}".rstrip(": ")]
    finally:
        if page is not None:
            try:
                page.close()
            except Exception:  # a crashed page may not close; the context does it
                pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--all-sizes-all-pages", action="store_true", help="every page at every size"
    )
    args = parser.parse_args()
    if not DOCS.is_dir():
        print("docs/ not found: run `quarto render` first")
        return 2
    every_page = sorted(str(p.relative_to(DOCS)) for p in DOCS.rglob("*.html"))
    server, base = serve(DOCS)
    failures = 0
    checked = 0
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            for n, (size, options) in enumerate(SIZES.items()):
                pages = every_page if n == 0 or args.all_sizes_all_pages else REPRESENTATIVE
                for theme in THEMES:
                    context = browser.new_context(**options, color_scheme=theme)
                    context.add_init_script(theme_script(theme))
                    for path in pages:
                        problems = run(check_page, context, base + path, theme)
                        checked += 1
                        for p in problems:
                            print(f"FAIL  {size} {theme} {path}: {p}")
                        failures += bool(problems)
                    problems = run(check_keyboard, context, base + KEYBOARD_PAGE)
                    checked += 1
                    for p in problems:
                        print(f"FAIL  {size} {theme} {KEYBOARD_PAGE} (keyboard): {p}")
                    failures += bool(problems)
                    context.close()
            browser.close()
    finally:
        server.shutdown()
    print(f"{checked} page loads checked, {failures} with problems")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
