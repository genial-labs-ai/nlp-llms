"""Build Workshop Purchases v1: data/purchases_v1.csv.gz, a synthetic retail purchases table.

The table behind the RFM (recency, frequency, monetary) app of Module 0
(agents-intro/rfm/). It is synthetic: no real customer is in it. Standard library only,
no network. Run from anywhere:

    python data/build_purchases.py          # write purchases_v1.csv.gz
    python data/build_purchases.py --check  # rebuild in memory; exit 1 if the file differs

One row per invoice line, sorted by date, then invoice:

    invoice_id,invoice_date,customer_id,country,stock_code,quantity,unit_price
    500001,2024-01-01,10052,United Kingdom,P0137,3,2.95

How a customer is drawn (the archetype is hidden: it is not in the file):

- Each customer belongs to one of six archetypes with fixed shares: `regular`,
  `loyal` (frequent, larger baskets), `occasional`, `wholesale` (rare, very large
  baskets), `lapsed` (bought regularly, then stopped early) and `new` (joined in the
  last 90 days). An archetype fixes ranges, not values.
- A first purchase date, an individual purchase rate (the archetype's rate times a
  log-normal factor, so customers of one archetype differ), and, for customers who
  churn, an exponential lifetime after which they never buy again.
- Invoices form a Poisson process between the first purchase and the end of activity.
  Each invoice has a few lines; each line picks a product from a fixed catalog of 200
  products (the product fixes the unit price) and a quantity.

So recency, frequency and spend vary continuously, and the archetypes overlap: a
clustering will find groups, not recover the archetypes exactly.

Determinism. Every random stream is `random.Random(f"{SEED}:<what>:<id>")`; Python
hashes string seeds with SHA-512, so the streams are the same on every platform and do
not depend on the order of the loop. The CSV uses "\\n" line endings and fixed number
formats. gzip uses `mtime=0` and no file name, so the same Python and zlib give the same
bytes; the compressed bytes can differ across zlib builds, so tests compare the
decompressed text.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import random
import sys
from datetime import date, timedelta
from pathlib import Path

SEED = 0
NAME = "Workshop Purchases v1"
HERE = Path(__file__).resolve().parent
OUT = HERE / "purchases_v1.csv.gz"

N_CUSTOMERS = 3000
FIRST_CUSTOMER_ID = 10001
FIRST_INVOICE_ID = 500001
START = date(2024, 1, 1)
END = date(2025, 12, 31)  # last possible invoice date; RFM uses the next day as "today"
DAYS = (END - START).days + 1
N_PRODUCTS = 200
COLUMNS = [
    "invoice_id",
    "invoice_date",
    "customer_id",
    "country",
    "stock_code",
    "quantity",
    "unit_price",
]

# Country shares, loosely shaped like a UK online retailer's customer base.
COUNTRIES = [
    ("United Kingdom", 0.70),
    ("Germany", 0.07),
    ("France", 0.06),
    ("Ireland", 0.04),
    ("Netherlands", 0.04),
    ("Spain", 0.03),
    ("Belgium", 0.02),
    ("Portugal", 0.02),
    ("Italy", 0.01),
    ("Australia", 0.01),
]

# share: fraction of customers. rate: invoices per 30 days for a typical member.
# churn: probability the customer stops buying; lifetime: mean days before stopping.
# lines and qty: inclusive ranges per invoice and per line. joins: window of first purchase.
ARCHETYPES = {
    "regular": dict(share=0.34, rate=0.45, churn=0.25, lifetime=300, lines=(1, 3), qty=(1, 6)),
    "loyal": dict(share=0.10, rate=1.60, churn=0.05, lifetime=500, lines=(2, 5), qty=(2, 10)),
    "occasional": dict(share=0.24, rate=0.10, churn=0.30, lifetime=300, lines=(1, 2), qty=(1, 4)),
    "wholesale": dict(share=0.05, rate=0.25, churn=0.10, lifetime=500, lines=(4, 9), qty=(12, 48)),
    "lapsed": dict(share=0.17, rate=0.70, churn=1.00, lifetime=120, lines=(1, 3), qty=(1, 6)),
    "new": dict(share=0.10, rate=0.90, churn=0.00, lifetime=0, lines=(1, 3), qty=(1, 6)),
}


def catalog() -> list[tuple[str, float]]:
    """200 products: (stock_code, unit_price). Prices are log-normal, median about 3."""
    rng = random.Random(f"{SEED}:catalog")
    return [
        (f"P{k:04d}", round(min(max(rng.lognormvariate(1.1, 0.8), 0.29), 60.0), 2))
        for k in range(1, N_PRODUCTS + 1)
    ]


def archetype_of(rng: random.Random) -> str:
    names = list(ARCHETYPES)
    return rng.choices(names, weights=[ARCHETYPES[n]["share"] for n in names])[0]


def customer_invoices(customer_id: int, products: list[tuple[str, float]]) -> list[dict]:
    """All invoices of one customer: [{"date", "customer_id", "country", "lines"}]."""
    rng = random.Random(f"{SEED}:customer:{customer_id}")
    kind = archetype_of(rng)
    a = ARCHETYPES[kind]
    country = rng.choices([c for c, _ in COUNTRIES], weights=[w for _, w in COUNTRIES])[0]

    if kind == "new":
        first = DAYS - 1 - rng.randrange(90)
    elif kind == "lapsed":
        first = rng.randrange(DAYS // 2)  # joined in the first year, so the lapse shows
    else:
        first = rng.randrange(DAYS - 30)
    last = DAYS - 1
    if rng.random() < a["churn"]:
        last = min(last, first + int(rng.expovariate(1 / a["lifetime"])))
    rate_per_day = a["rate"] / 30 * rng.lognormvariate(0.0, 0.5)
    # Each customer favors a subset of the catalog.
    favorites = rng.sample(products, 25)

    invoices = []
    day = float(first)
    while day <= last:
        lines = []
        for _ in range(rng.randint(*a["lines"])):
            code, price = rng.choice(favorites)
            lines.append((code, rng.randint(*a["qty"]), price))
        invoices.append(
            {
                "date": START + timedelta(days=int(day)),
                "customer_id": customer_id,
                "country": country,
                "lines": lines,
            }
        )
        day += rng.expovariate(rate_per_day)
    return invoices


def build() -> tuple[str, bytes]:
    """Return (csv_text, gzip_bytes)."""
    products = catalog()
    invoices = []
    for k in range(N_CUSTOMERS):
        invoices.extend(customer_invoices(FIRST_CUSTOMER_ID + k, products))
    invoices.sort(key=lambda inv: (inv["date"], inv["customer_id"]))

    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(COLUMNS)
    for n, inv in enumerate(invoices):
        for code, qty, price in inv["lines"]:
            writer.writerow(
                [
                    FIRST_INVOICE_ID + n,
                    inv["date"].isoformat(),
                    inv["customer_id"],
                    inv["country"],
                    code,
                    qty,
                    f"{price:.2f}",
                ]
            )
    text = buf.getvalue()
    out = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=out, mtime=0, compresslevel=9) as f:
        f.write(text.encode("ascii"))
    return text, out.getvalue()


def summary(text: str) -> str:
    rows = list(csv.DictReader(io.StringIO(text)))
    customers = {r["customer_id"] for r in rows}
    invoices = {r["invoice_id"] for r in rows}
    return f"{len(rows):,} lines, {len(invoices):,} invoices, {len(customers):,} customers"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="compare with the committed file")
    args = parser.parse_args()
    text, blob = build()
    if args.check:
        if not OUT.exists():
            print(f"{OUT.name} does not exist")
            return 1
        committed = OUT.read_bytes()
        if gzip.decompress(committed).decode("ascii") != text:
            print(f"{OUT.name}: content differs from a rebuild")
            return 1
        same = "same" if committed == blob else "different (zlib build), same content"
        print(f"{OUT.name}: content matches; compressed bytes {same}")
        return 0
    OUT.write_bytes(blob)
    print(f"Wrote {OUT.name}: {summary(text)}")
    print(f"  {len(blob):,} bytes, sha256 {hashlib.sha256(blob).hexdigest()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
