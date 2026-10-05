"""Render a Module 0 app headlessly and check that its three.js scene draws.

Instructor tool, not part of the participant's task. It serves one app directory over
HTTP, opens index.html in headless Chromium with Playwright, and answers every request
for the pinned three.js files on jsDelivr from a local copy of the same npm release, so
it works without CDN access. The local copy is downloaded once from the npm registry
into agents-intro/.cache/ (gitignored) and checked against the registry's sha512.

    python agents-intro/check_render.py agents-intro/protein/python --out /tmp/shots
    python agents-intro/check_render.py agents-intro/rfm/r --out /tmp/shots

Needs `pip install playwright` and a Chromium (`playwright install chromium`, or set
CHROMIUM_PATH). The app's data.json must exist (run its build first).

Checks: the page sets `window.__ready`; no console errors and no page errors; the
WebGL canvas is not blank (many distinct colors, a large share of pixels differ from
the background); one real mouse interaction works (a click selects a residue, or a
hover shows a customer). Saves two screenshots per app. Exits 1 on any failure.
"""

from __future__ import annotations

import argparse
import base64
import functools
import hashlib
import http.server
import io
import json
import mimetypes
import os
import re
import sys
import tarfile
import threading
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE = HERE / ".cache"
CDN = re.compile(r"https://cdn\.jsdelivr\.net/npm/three@(?P<version>[\d.]+)/")


def three_version(html: str) -> str:
    versions = set(CDN.findall(html))
    if len(versions) != 1:
        raise SystemExit(f"Expected one pinned three.js version in index.html, found {versions}")
    return versions.pop()


def local_three(version: str) -> Path:
    """Return the `package/` directory of three@version, downloading it if needed."""
    root = CACHE / f"three-{version}"
    if (root / "package" / "build" / "three.module.js").exists():
        return root / "package"
    meta_url = f"https://registry.npmjs.org/three/{version}"
    with urllib.request.urlopen(meta_url, timeout=60) as response:
        dist = json.load(response)["dist"]
    with urllib.request.urlopen(dist["tarball"], timeout=120) as response:
        blob = response.read()
    algorithm, expected = dist["integrity"].split("-", 1)
    if base64.b64encode(hashlib.new(algorithm, blob).digest()).decode() != expected:
        raise SystemExit(f"{dist['tarball']} does not match the registry's {algorithm}")
    root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(blob)) as tar:
        tar.extractall(root, filter="data")
    print(f"Downloaded three@{version} to {root}")
    return root / "package"


def serve(directory: Path) -> tuple[http.server.ThreadingHTTPServer, int]:
    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass

    handler = functools.partial(Quiet, directory=str(directory))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, server.server_address[1]


def canvas_stats(png: bytes) -> tuple[int, float]:
    """(distinct colors, share of pixels that differ from the corner pixel)."""
    from PIL import Image

    image = Image.open(io.BytesIO(png)).convert("RGB")
    pixels = list(image.getdata())  # Pillow 12 deprecates getdata(); fine until Pillow 14
    background = pixels[0]
    differ = sum(
        1 for p in pixels if sum(abs(a - b) for a, b in zip(p, background, strict=True)) > 24
    )
    return len(set(pixels)), differ / len(pixels)


def check(app: Path, out: Path, chromium: str | None) -> list[str]:
    from playwright.sync_api import sync_playwright

    html = (app / "index.html").read_text()
    if not (app / "data.json").exists():
        return [f"{app}/data.json does not exist: run the app's build first"]
    version = three_version(html)
    package = local_three(version)
    server, port = serve(app)
    name = "-".join(app.resolve().parts[-2:])
    problems: list[str] = []
    messages: list[str] = []

    def route_cdn(route):
        rel = CDN.sub("", route.request.url).split("?")[0]
        path = package / rel
        if not path.is_file():
            route.fulfill(status=404, body=f"not in local three@{version}: {rel}")
            return
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        route.fulfill(status=200, body=path.read_bytes(), headers={"Content-Type": mime})

    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            executable_path=chromium,
            args=[
                "--use-angle=swiftshader",
                "--enable-unsafe-swiftshader",
                "--ignore-gpu-blocklist",
            ],
        )
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.route(re.compile(r"https://cdn\.jsdelivr\.net/npm/three@.*"), route_cdn)

        def on_console(message):
            if message.type in ("error", "warning"):
                messages.append(f"{message.type}: {message.text}")

        page.on("console", on_console)
        page.on("pageerror", lambda e: messages.append(f"pageerror: {e}"))
        page.goto(f"http://127.0.0.1:{port}/index.html")
        try:
            page.wait_for_function("window.__ready === true", timeout=20_000)
        except Exception:  # noqa: BLE001 - report whatever the page said
            problems.append("window.__ready was never set")
        page.wait_for_timeout(800)

        canvas = page.locator("canvas").first
        colors, share = canvas_stats(canvas.screenshot())
        print(f"{name}: canvas has {colors} distinct colors, {share:.1%} non-background pixels")
        if colors < 50 or share < 0.02:
            problems.append(f"canvas looks blank ({colors} colors, {share:.1%} drawn)")
        page.screenshot(path=str(out / f"{name}-1-loaded.png"))

        if not problems:
            problems += interact(page, name, out)

        browser.close()
    server.shutdown()
    errors = [m for m in messages if not m.startswith("warning")]
    for m in messages:
        print(f"  console {m}")
    problems += [f"console {m}" for m in errors]
    return problems


def interact(page, name: str, out: Path) -> list[str]:
    """One real mouse interaction per app type, then a second screenshot."""
    kind = page.evaluate("window.__viewer.kind")
    if kind == "protein":
        # Click residue Ile 44 (index 43), the center of ubiquitin's hydrophobic patch.
        point = page.evaluate("window.__viewer.screenPoint(43)")
        page.mouse.click(point["x"], point["y"])
        page.wait_for_timeout(300)
        selected = page.evaluate("window.__selected")
        page.select_option("#colorBy", "contacts")
        page.wait_for_timeout(300)
        page.screenshot(path=str(out / f"{name}-2-clicked.png"))
        details = page.inner_text("#details")
        print(f"  clicked residue index {selected}: {details.splitlines()[:1]}")
        return [] if selected is not None and details.strip() else ["click selected nothing"]
    if kind == "rfm":
        k = page.evaluate("window.__viewer.n") // 2
        point = page.evaluate(f"window.__viewer.screenPoint({k})")
        page.mouse.move(point["x"], point["y"])
        page.wait_for_timeout(300)
        tip = page.inner_text("#tooltip") if page.is_visible("#tooltip") else ""
        page.screenshot(path=str(out / f"{name}-2-hover.png"))
        print(f"  hover tooltip: {' | '.join(tip.splitlines())}")
        return [] if tip.strip() else ["hover showed no tooltip"]
    return [f"unknown viewer kind {kind!r}"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("apps", nargs="+", type=Path, help="app directories with index.html")
    parser.add_argument("--out", type=Path, default=Path("."), help="screenshot directory")
    parser.add_argument("--chromium", default=os.environ.get("CHROMIUM_PATH"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    failed = False
    for app in args.apps:
        problems = check(app, args.out, args.chromium)
        for p in problems:
            print(f"  FAIL {p}")
        print(f"{app}: {'FAIL' if problems else 'ok'}")
        failed |= bool(problems)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
