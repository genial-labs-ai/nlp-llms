"""Build data/arxiv_topics_v1.csv.gz, the classification set of the running thread.

Source: the arXiv API (https://info.arxiv.org/help/api/). arXiv metadata is
CC0 1.0. This script is kept for provenance; notebooks never run it. They load
the committed file (see data/README.md).

    uv run --group site python data/build_arxiv_topics.py

Recipe, all of it fixed:

- For each category in CATEGORIES, page through every e-print submitted in
  each month of MONTHS, oldest first, and keep those whose *primary* category
  is that one. (One query per month: the API returns errors when a single
  query is paged past 10,000 results.)
- Sort the kept records by arXiv ID, shuffle with random.Random(SEED), and take
  the first PER_CLASS. The first 1,200 are train, the next 150 validation and
  the last 400 test.
- Write the rows grouped by split (train, val, test) and shuffled within each
  split with the same seed.

The raw API pages are cached under data/.cache/ (gitignored) so a rebuild does
not hit the API again. arXiv asks for at most one request every three seconds.
"""

import csv
import gzip
import hashlib
import io
import json
import random
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

DATA = Path(__file__).resolve().parent
CACHE = DATA / ".cache"
OUT = DATA / "arxiv_topics_v1.csv.gz"

CATEGORIES = ["cs.CL", "cs.CV", "cs.LG", "cs.RO"]  # label = index in this list
# Submission windows, one query each: 2024-01-01 to 2024-06-30.
MONTHS = [
    ("20240101", "20240131"),
    ("20240201", "20240229"),
    ("20240301", "20240331"),
    ("20240401", "20240430"),
    ("20240501", "20240531"),
    ("20240601", "20240630"),
]
SEED = 0
SPLITS = [("train", 1200), ("val", 150), ("test", 400)]  # per class
PER_CLASS = sum(n for _, n in SPLITS)
PAGE = 1000
NS = {"a": "http://www.w3.org/2005/Atom", "x": "http://arxiv.org/schemas/atom"}


def fetch_page(cat: str, month: tuple[str, str], start: int) -> list[dict]:
    """One API page as a list of records (all categories, cross-lists included)."""
    cached = CACHE / f"{cat}-{month[0]}-{start:06d}.json"
    if cached.exists():
        return json.loads(cached.read_text(encoding="utf-8"))
    query = urllib.parse.urlencode(
        {
            "search_query": f"cat:{cat} AND submittedDate:[{month[0]}0000 TO {month[1]}2359]",
            "start": start,
            "max_results": PAGE,
            "sortBy": "submittedDate",
            "sortOrder": "ascending",
        }
    )
    req = urllib.request.Request(
        "https://export.arxiv.org/api/query?" + query,
        headers={"User-Agent": "nlp-llms-workshop dataset build"},
    )
    for attempt in range(5):
        time.sleep(3.1 + 5 * attempt)
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                root = ET.fromstring(r.read())
        except OSError as e:
            print(f"  retry {cat} {month[0]} {start}: {e}", flush=True)
            continue
        total = int(root.find("{http://a9.com/-/spec/opensearch/1.1/}totalResults").text)
        entries = root.findall("a:entry", NS)
        if not entries and start < total:
            print(f"  retry {cat} {month[0]} {start}: empty page, total {total}", flush=True)
            continue
        rows = [
            {
                "id": e.find("a:id", NS).text.rsplit("/", 1)[-1],
                "primary": e.find("x:primary_category", NS).attrib["term"],
                "title": " ".join(e.find("a:title", NS).text.split()),
                "abstract": " ".join(e.find("a:summary", NS).text.split()),
            }
            for e in entries
        ]
        CACHE.mkdir(exist_ok=True)
        cached.write_text(json.dumps(rows), encoding="utf-8")
        return rows
    raise RuntimeError(f"arXiv API kept failing for {cat} {month[0]} at start={start}")


def fetch_category(cat: str) -> list[dict]:
    kept = {}
    for month in MONTHS:
        start = 0
        while True:
            rows = fetch_page(cat, month, start)
            if not rows:
                break
            for r in rows:
                if r["primary"] == cat:
                    kept[r["id"]] = r
            start += PAGE
        print(f"{cat}: through {month[1]}, {len(kept)} with this primary category", flush=True)
    return [kept[k] for k in sorted(kept)]


def main() -> None:
    rng = random.Random(SEED)
    by_split = {name: [] for name, _ in SPLITS}
    for label, cat in enumerate(CATEGORIES):
        rows = fetch_category(cat)
        if len(rows) < PER_CLASS:
            raise RuntimeError(f"{cat}: only {len(rows)} records, need {PER_CLASS}")
        rng.shuffle(rows)
        at = 0
        for name, n in SPLITS:
            for r in rows[at : at + n]:
                by_split[name].append([name, label, cat, r["id"], r["title"], r["abstract"]])
            at += n
    text = io.StringIO()
    writer = csv.writer(text, lineterminator="\n")
    writer.writerow(["split", "label", "category", "arxiv_id", "title", "abstract"])
    for name, _ in SPLITS:
        rng.shuffle(by_split[name])
        writer.writerows(by_split[name])
    raw = text.getvalue().encode("utf-8")
    # mtime=0 and no filename in the header: the same input gives the same bytes.
    with open(OUT, "wb") as f, gzip.GzipFile(fileobj=f, mode="wb", mtime=0, filename="") as gz:
        gz.write(raw)
    blob = OUT.read_bytes()
    print(f"{OUT.name}: {len(blob):,} bytes ({len(raw):,} uncompressed)")
    print("sha256", hashlib.sha256(blob).hexdigest())


if __name__ == "__main__":
    main()
