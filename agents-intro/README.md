# Module 0 reference solutions

Instructor material for Module 0, "Coding agents in the terminal" (`lectures/00-coding-agents.qmd`). Participants build these two apps themselves with a coding agent, in Python or R; these references show that the page's prompts lead somewhere that works, and give the numbers to compare against ([MEASURED.md](MEASURED.md)).

Each app is one script that writes `data.json`, plus an `index.html` that reads it and draws a three.js scene. Both language versions write the same fields, so `python/index.html` and `r/index.html` are the same file (a test keeps them identical). three.js is pinned to `agents_intro.threejs` in `_variables.yml` and loaded from jsDelivr through an import map.

```text
agents-intro/
├── protein/            ubiquitin (PDB 1UBQ): CA backbone, contacts, radius of gyration, hydrophobicity
│   ├── python/         build.py (NumPy, plain PDB parsing) + index.html
│   └── r/              build.R (bio3d, jsonlite) + index.html
├── rfm/                customer segments: RFM, log + z-score, k-means, named segments
│   ├── python/         build.py (pandas, scikit-learn) + index.html
│   └── r/              build.R (dplyr, base kmeans(), cluster::silhouette) + index.html
├── check_render.py     headless render check (Playwright), for instructors
└── MEASURED.md         the numbers the references produce
```

## Run

From the repository root. Each build prints its numbers and writes `data.json` next to its `index.html`; `data.json` is gitignored. Then serve the folder and open <http://localhost:8000> (the page loads `data.json` with `fetch`, which does not work from `file://`).

| App | Build | View |
|---|---|---|
| Protein, Python | `python agents-intro/protein/python/build.py` | `python -m http.server -d agents-intro/protein/python 8000` |
| Protein, R | `Rscript agents-intro/protein/r/build.R` | `python -m http.server -d agents-intro/protein/r 8000` |
| RFM, Python | `python agents-intro/rfm/python/build.py` | `python -m http.server -d agents-intro/rfm/python 8000` |
| RFM, R | `Rscript agents-intro/rfm/r/build.R` | `python -m http.server -d agents-intro/rfm/r 8000` |

Requirements: Python 3.12 with `numpy` (protein) or `numpy pandas scikit-learn` (RFM); R 4.3 with `bio3d jsonlite` (protein) or `dplyr jsonlite` (RFM; `cluster` ships with R). On Ubuntu 24.04: `apt-get install r-base-core r-cran-bio3d r-cran-jsonlite r-cran-dplyr r-cran-cluster`.

Data. The protein builds try RCSB (`files.rcsb.org`) first, then this repository's copy on GitHub, then `data/1ubq.pdb` in a clone; the Python build accepts a download only if its SHA-256 matches the recorded file. The RFM builds read `data/purchases_v1.csv.gz` in a clone, else download it from this repository. Pass a file to use a local copy: `build.py --pdb data/1ubq.pdb`, `build.py --csv data/purchases_v1.csv.gz`, or `build.R <file>`. Both datasets are described in [data/README.md](../data/README.md#module-0-sets-ubiquitin-and-purchases).

Publishing on GitHub Pages (in a participant's own repository): commit the app folder including `data.json`, then serve it from Pages. The pages use only relative paths, so they work under a project subpath.

## Headless check

`check_render.py` serves an app folder, opens it in headless Chromium, answers the jsDelivr three.js requests from a local copy of the same npm release (downloaded once into `agents-intro/.cache/`, gitignored, and checked against the registry's sha512), and fails on console errors, a blank canvas, or a click or hover that does nothing. It saves two screenshots per app.

```text
pip install playwright pillow && playwright install chromium
python agents-intro/check_render.py agents-intro/protein/python agents-intro/rfm/r --out /tmp/shots
```

Set `CHROMIUM_PATH` to use an existing Chromium instead of Playwright's.
