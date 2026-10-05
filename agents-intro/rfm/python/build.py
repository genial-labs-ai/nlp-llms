"""Customer segments from purchases: RFM, k-means, and data.json for a 3D scatter.

Reference solution for Module 0, app 2, in Python (pandas + scikit-learn). Writes
data.json next to this script for index.html to render with three.js.

    python agents-intro/rfm/python/build.py
    python agents-intro/rfm/python/build.py --csv data/purchases_v1.csv.gz
    python -m http.server -d agents-intro/rfm/python 8000      # then open :8000

Steps, and the choices made:

1. RFM per customer, as of SNAPSHOT (2026-01-01, the day after the last invoice):
   recency = days since the customer's last invoice (1 = bought on the last day),
   frequency = number of distinct invoices, monetary = total of quantity * unit_price.
2. Scale: log1p of each (all three are right-skewed; a few big customers would
   dominate the distances otherwise), then standardize to mean 0, standard deviation 1
   (population SD, ddof = 0, so R and Python agree).
3. k-means (k-means++ start, 20 restarts, seed 0) for k = 3, 4, 5, 6; keep the k with
   the highest mean silhouette. k = 2 is not considered: two groups are too coarse to
   act on.
4. Name each segment from its centroid in standardized units (zR, zF, zM). Recent means
   zR < 0 (fewer days than average); value = (zF + zM) / 2:
       recent, value > 0.75   -> Champions
       recent, value > 0      -> Loyal
       recent, value <= 0     -> New or promising
       not recent, value > 0  -> At risk
       not recent, value <= 0 -> Lost
   If two segments get the same name, the one with the higher value keeps it and the
   next gets " (2)".
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

HERE = Path(__file__).resolve().parent
OUT = HERE / "data.json"
REPO_COPY = HERE.parents[2] / "data" / "purchases_v1.csv.gz"  # present inside a clone

# The same entry as `datasets.purchases` in _variables.yml (tests check they agree).
URLS = [
    "https://raw.githubusercontent.com/project-delphi/nlp-llms/main/data/purchases_v1.csv.gz",
    "https://cdn.jsdelivr.net/gh/project-delphi/nlp-llms@main/data/purchases_v1.csv.gz",
]
SHA256 = "ac73eac9219793d280961ed0e60e1a1649ece3e8ab7c8624da4a4928629fa11b"

SNAPSHOT = pd.Timestamp("2026-01-01")
K_RANGE = range(3, 7)
SEED = 0
COLORS = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#b07aa1", "#edc948"]


def load_purchases(path: str | None) -> tuple[pd.DataFrame, str]:
    """The purchases table: the local copy in a clone, else the repository URLs."""
    if path:
        return pd.read_csv(path, parse_dates=["invoice_date"]), Path(path).name
    if REPO_COPY.exists():
        return pd.read_csv(REPO_COPY, parse_dates=["invoice_date"]), "data/purchases_v1.csv.gz"
    for url in URLS:
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                blob = response.read()
        except OSError as error:
            print(f"Could not fetch {url}: {error}")
            continue
        if hashlib.sha256(blob).hexdigest() != SHA256:
            print(f"Ignoring {url}: it does not match the recorded file")
            continue
        frame = pd.read_csv(io.BytesIO(blob), compression="gzip", parse_dates=["invoice_date"])
        return frame, url
    raise RuntimeError("Could not load the purchases table")


def rfm_table(purchases: pd.DataFrame) -> pd.DataFrame:
    """One row per customer: recency (days), frequency (invoices), monetary (spend)."""
    purchases = purchases.assign(amount=purchases["quantity"] * purchases["unit_price"])
    rfm = purchases.groupby("customer_id").agg(
        last=("invoice_date", "max"),
        frequency=("invoice_id", "nunique"),
        monetary=("amount", "sum"),
    )
    rfm["recency"] = (SNAPSHOT - rfm["last"]).dt.days
    return rfm[["recency", "frequency", "monetary"]].sort_index()


def standardize(rfm: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """log1p, then z-scores with the population SD. Returns (z, mean, sd)."""
    logged = np.log1p(rfm.to_numpy(dtype=float))  # (customers, 3)
    mean, sd = logged.mean(axis=0), logged.std(axis=0)
    return (logged - mean) / sd, mean, sd


def choose_k(z: np.ndarray) -> tuple[int, dict[int, float]]:
    scores = {}
    for k in K_RANGE:
        labels = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit_predict(z)
        scores[k] = float(silhouette_score(z, labels))
    return max(scores, key=scores.get), scores


def name_segments(centroids_z: np.ndarray) -> list[str]:
    """Readable names from standardized centroids, sorted by value, highest first."""
    names: list[str] = []
    for z_r, z_f, z_m in centroids_z:
        value = (z_f + z_m) / 2
        if z_r < 0:
            name = "Champions" if value > 0.75 else "Loyal" if value > 0 else "New or promising"
        else:
            name = "At risk" if value > 0 else "Lost"
        repeats = sum(1 for n in names if n.split(" (")[0] == name)
        names.append(f"{name} ({repeats + 1})" if repeats else name)
    return names


def build(purchases: pd.DataFrame) -> dict:
    rfm = rfm_table(purchases)
    z, mean, sd = standardize(rfm)
    k, scores = choose_k(z)
    model = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit(z)

    # Order segments by value, highest first, so ids are stable and readable.
    centroids = model.cluster_centers_  # (k, 3)
    order = np.argsort(-(centroids[:, 1] + centroids[:, 2]))
    relabel = {old: new for new, old in enumerate(order)}
    labels = np.array([relabel[c] for c in model.labels_])
    centroids = centroids[order]
    names = name_segments(centroids)
    raw_centroids = np.expm1(centroids * sd + mean)  # back to days, invoices, spend

    segments = []
    for s in range(k):
        segments.append(
            {
                "id": s,
                "name": names[s],
                "color": COLORS[s],
                "count": int((labels == s).sum()),
                "centroid": {
                    "recency": round(float(raw_centroids[s, 0]), 1),
                    "frequency": round(float(raw_centroids[s, 1]), 2),
                    "monetary": round(float(raw_centroids[s, 2]), 2),
                },
                "centroid_z": [round(float(v), 4) for v in centroids[s]],
            }
        )
    customers = [
        {
            "id": int(cid),
            "recency": int(row.recency),
            "frequency": int(row.frequency),
            "monetary": round(float(row.monetary), 2),
            "z": [round(float(v), 4) for v in z[n]],
            "segment": int(labels[n]),
        }
        for n, (cid, row) in enumerate(rfm.iterrows())
    ]
    meta = {
        "snapshot": SNAPSHOT.date().isoformat(),
        "n_customers": len(rfm),
        "n_invoices": int(purchases["invoice_id"].nunique()),
        "k": k,
        "k_rule": "highest mean silhouette over k = 3..6 (k-means, 20 restarts, seed 0)",
        "silhouette": {str(key): round(v, 4) for key, v in scores.items()},
        "transform": "z = (log1p(x) - mean) / sd, population sd",
        "log_mean": [round(float(v), 6) for v in mean],
        "log_sd": [round(float(v), 6) for v in sd],
        "made_by": "python",
    }
    return {"meta": meta, "segments": segments, "customers": customers}


def main() -> None:
    parser = argparse.ArgumentParser(description="RFM segments for index.html.")
    parser.add_argument("--csv", help="read this purchases file instead of the default")
    parser.add_argument("--out", default=str(OUT), help="where to write data.json")
    args = parser.parse_args()

    purchases, source = load_purchases(args.csv)
    data = build(purchases)
    data["meta"]["source"] = source
    Path(args.out).write_text(json.dumps(data, separators=(",", ":")) + "\n")
    m = data["meta"]
    print(f"Read {source}: {len(purchases):,} lines, {m['n_customers']:,} customers")
    print(f"Silhouette by k: {m['silhouette']} -> k = {m['k']}")
    for s in data["segments"]:
        c = s["centroid"]
        print(
            f"  {s['name']:<18} {s['count']:>5} customers   centroid: {c['recency']:>6.1f} days, "
            f"{c['frequency']:>5.2f} invoices, {c['monetary']:>8.2f} spend"
        )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
