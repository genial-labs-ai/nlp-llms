# Measured numbers

What the four references printed on 2026-10-05, run from the repository root on the build container (Linux, 4-core Intel Xeon at 2.10 GHz). Python 3.12.3 with NumPy 2.5.3, pandas 3.0.6 and scikit-learn 1.9.1; R 4.3.3 with bio3d 2.4.4, jsonlite 1.8.8, dplyr 1.1.4 and cluster 2.1.6. RCSB and GitHub were not reachable from the container for these files, so every build read the committed copies (`data/1ubq.pdb`, SHA-256 `d4a6812d…7b161`; `data/purchases_v1.csv.gz`, SHA-256 `ac73eac9…fa11b`). `tests/test_agents_intro.py` asserts the numbers marked as tested.

## Protein: ubiquitin, PDB 1UBQ

Python and R wrote identical residues and contacts (every field equal).

| Quantity | Value | Definition |
|---|---|---|
| Residues | 76 | one CA atom each, `MQIFVKTLTG…LRLRGG` |
| Heavy atoms | 602 | all `ATOM` records; the file has no hydrogens; 58 waters excluded |
| Radius of gyration, CA | **11.493 Å** (tested) | unweighted, the 76 CA atoms |
| Radius of gyration, heavy atoms | **11.726 Å** (tested) | unweighted, all 602 protein heavy atoms |
| Radius of gyration, heavy atoms, mass-weighted | 11.788 Å (tested) | C 12.011, N 14.007, O 15.999, S 32.06 |
| Contacts | **177** (tested) | residue pairs with CA–CA ≤ 8.0 Å and \|i − j\| ≥ 3 |
| Contacts per residue | 0 to 9 | most: Ile 3, Ile 23, Leu 56 (9 each); none: Thr 9, Ala 46, Gly 75 |
| B-factor (CA) | 3.51 to 36.19 Å² | highest at the C-terminal tail: Gly 76, Gly 75, Arg 74, Leu 73 |
| Kyte–Doolittle | −4.5 to +4.5 | the scale's own range; no window averaging |

Why \|i − j\| ≥ 3: consecutive CA atoms are always 3.8 Å apart (measured 3.76–3.85 Å) and CA atoms two apart are at most about 7.3 Å (measured maximum 7.17 Å), so pairs with \|i − j\| < 3 are within 8 Å whatever the fold and say nothing about it. The same cutoff with other separations, for comparison: \|i − j\| ≥ 1 gives 326 pairs, ≥ 2 gives 251, ≥ 6 gives 116, ≥ 12 gives 92, ≥ 24 gives 73.

## RFM: Workshop Purchases v1

48,953 invoice lines, 18,432 invoices, 3,000 customers; RFM as of 2026-01-01. Features: log1p, then z-scores with the population SD. k-means with 20 restarts, seed 0.

Mean silhouette by k (k = 2 is not considered):

| k | Python (scikit-learn) | R (`kmeans`, `cluster::silhouette`) |
|---|---|---|
| 3 | 0.3385 | 0.3386 |
| **4** | **0.3475** | **0.3475** |
| 5 | 0.3241 | 0.3250 |
| 6 | 0.3063 | 0.3424 |

Both choose **k = 4** (tested). The margin is small: 0.009 over k = 3 in Python, and only 0.005 over k = 6 in R, where `kmeans()` found a better six-cluster solution than scikit-learn did. A participant's run with a different library, seed or number of restarts can reasonably land on another k; that is a point to discuss, not an error.

Segments, ordered by value; centroids back-transformed to original units (so they are geometric-mean-like, not arithmetic means):

| Segment | Python count | R count | Centroid recency | Centroid frequency | Centroid spend |
|---|---|---|---|---|---|
| Champions | **617** | 617 | 17.7 days | 14.1 invoices | 973.69 |
| At risk | **789** | 790 | 181.0 days | 4.9 invoices | 201.18 |
| New or promising | **644** | 644 | 18.4 days | 2.7 invoices | 60.43 |
| Lost | **950** | 949 | 280.9 days | 1.4 invoices | 21.40 |

Centroids are Python's; R's differ by less than 0.2% (for example Champions 972.31 spend, At risk 181.2 days and 200.92 spend). The test allows each count to move by 5 customers across library versions. The naming rule: recent means zR < 0; value = (zF + zM) / 2; recent with value > 0.75 is Champions, recent with value > 0 Loyal, other recent New or promising, not recent with value > 0 At risk, the rest Lost. No segment is named Loyal at k = 4.

## Run times

Wall clock, including interpreter start, on the container above: protein Python 0.9 s, protein R 1.0 s, RFM Python 3.1 s, RFM R 3.4 s. Building the purchases table (`data/build_purchases.py`) takes 0.6 s.
