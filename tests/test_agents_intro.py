"""Checks on the Module 0 reference solutions (agents-intro/) and their data.

No network. The structural checks need only the standard library and PyYAML. The
protein check needs NumPy and the RFM checks need pandas and scikit-learn; they skip
cleanly when those are missing (the CI test job has neither; the notebooks job has both).
If `Rscript` with bio3d, jsonlite, dplyr and cluster is on PATH, the R references are
run too, into a temporary directory.

The recorded numbers below are quoted in agents-intro/MEASURED.md and the Module 0 page.
"""

import gzip
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

ROOT = Path(__file__).resolve().parent.parent
APPS = ROOT / "agents-intro"
DATA = ROOT / "data"
V = yaml.safe_load((ROOT / "_variables.yml").read_text(encoding="utf-8"))
DATASETS = V["datasets"]

# Recorded on 2026-10-05 from the committed data/1ubq.pdb and data/purchases_v1.csv.gz.
PROTEIN = {
    "n_residues": 76,
    "n_heavy_atoms": 602,
    "n_contacts": 177,  # CA-CA <= 8.0 A, |i - j| >= 3
    "rg_ca_angstrom": 11.493,
    "rg_heavy_angstrom": 11.726,
    "rg_heavy_mass_weighted_angstrom": 11.788,
}
RFM_K = 4
RFM_SEGMENTS = {"Champions": 617, "At risk": 789, "New or promising": 644, "Lost": 950}
RFM_TOLERANCE = 5  # customers; k-means can move a few border points across library versions

HAVE_NUMPY = importlib.util.find_spec("numpy") is not None
HAVE_RFM_STACK = all(importlib.util.find_spec(m) for m in ("numpy", "pandas", "sklearn"))


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def required_fields(app: str) -> dict:
    """The REQUIRED field list that the app's index.html checks data.json against."""
    html = (APPS / app / "python" / "index.html").read_text(encoding="utf-8")
    return json.loads(re.search(r"const REQUIRED = (\{.*?\});", html).group(1))


def assert_fields(test: unittest.TestCase, data: dict, required: dict) -> None:
    for key, fields in required.items():
        test.assertIn(key, data)
        value = data[key]
        if isinstance(value, list):
            test.assertTrue(value, key)
            items = value if key != "contacts" else []
        else:
            items = [value]
        for item in items:
            for field in fields:
                test.assertIn(field, item, f"{key}.{field}")


def python_constant(path: Path, name: str):
    """A string or list-of-strings constant from a Python source file, without importing it."""
    source = path.read_text(encoding="utf-8")
    match = re.search(rf"^{name} = (\[.*?\]|\".*?\")", source, flags=re.MULTILINE | re.DOTALL)
    return eval(match.group(1))  # noqa: S307 - our own literal


def r_urls(path: Path) -> list[str]:
    block = re.search(r"urls <- c\((.*?)\)", path.read_text(encoding="utf-8"), re.DOTALL).group(1)
    return re.findall(r'"(https://[^"]+)"', block)


def rscript_ready() -> bool:
    if not shutil.which("Rscript"):
        return False
    check = "for (p in c('bio3d','jsonlite','dplyr','cluster')) stopifnot(requireNamespace(p))"
    return subprocess.run(["Rscript", "-e", check], capture_output=True).returncode == 0


class Pages(unittest.TestCase):
    def test_python_and_r_pages_are_identical(self):
        for app in ("protein", "rfm"):
            python = (APPS / app / "python" / "index.html").read_bytes()
            r = (APPS / app / "r" / "index.html").read_bytes()
            self.assertEqual(python, r, f"{app}: copy python/index.html to r/index.html")

    def test_three_js_is_pinned_to_the_variables_version(self):
        version = V["agents_intro"]["threejs"]
        for page in APPS.glob("*/*/index.html"):
            html = page.read_text(encoding="utf-8")
            importmap = json.loads(
                re.search(r'<script type="importmap">(.*?)</script>', html, re.DOTALL).group(1)
            )["imports"]
            self.assertEqual(
                importmap["three"],
                f"https://cdn.jsdelivr.net/npm/three@{version}/build/three.module.js",
                page,
            )
            self.assertEqual(
                importmap["three/addons/"],
                f"https://cdn.jsdelivr.net/npm/three@{version}/examples/jsm/",
                page,
            )
            self.assertEqual(set(re.findall(r"three@([\d.]+)", html)), {version}, page)

    def test_required_field_lists_parse(self):
        self.assertIn("rg_ca_angstrom", required_fields("protein")["meta"])
        self.assertIn("segment", required_fields("rfm")["customers"])


class Sources(unittest.TestCase):
    """The references fetch the same URLs and hashes as _variables.yml records."""

    def test_protein_sources(self):
        d = DATASETS["ubiquitin"]
        script = APPS / "protein" / "python" / "build.py"
        self.assertEqual(python_constant(script, "URLS"), d["urls"])
        self.assertEqual(python_constant(script, "SHA256"), d["sha256"])
        self.assertEqual(r_urls(APPS / "protein" / "r" / "build.R"), d["urls"])
        self.assertTrue(d["urls"][0].startswith("https://files.rcsb.org/"))

    def test_rfm_sources(self):
        d = DATASETS["purchases"]
        script = APPS / "rfm" / "python" / "build.py"
        self.assertEqual(python_constant(script, "URLS"), d["urls"])
        self.assertEqual(python_constant(script, "SHA256"), d["sha256"])
        self.assertEqual(r_urls(APPS / "rfm" / "r" / "build.R"), d["urls"])

    def test_committed_pdb_is_the_recorded_entry(self):
        text = (DATA / "1ubq.pdb").read_text(encoding="ascii")
        self.assertTrue(text.startswith("HEADER    CHROMOSOMAL PROTEIN"))
        self.assertIn("1UBQ", text.splitlines()[0])
        lines = text.splitlines()
        self.assertEqual(sum(line.startswith("ATOM") for line in lines), 602)
        self.assertEqual(sum(line.startswith("HETATM") and " HOH " in line for line in lines), 58)
        self.assertIn("REVDAT   6   14-FEB-24 1UBQ", text)


class Purchases(unittest.TestCase):
    def test_rebuild_gives_the_committed_content(self):
        builder = load_module(DATA / "build_purchases.py", "build_purchases")
        text, _ = builder.build()
        committed = gzip.decompress((DATA / "purchases_v1.csv.gz").read_bytes()).decode("ascii")
        self.assertEqual(text, committed)

    def test_shape_matches_variables(self):
        d = DATASETS["purchases"]
        rows = gzip.decompress((DATA / "purchases_v1.csv.gz").read_bytes()).decode().splitlines()
        self.assertEqual(
            rows[0], "invoice_id,invoice_date,customer_id,country,stock_code,quantity,unit_price"
        )
        body = [r.split(",") for r in rows[1:]]
        self.assertEqual(len(body), d["lines"])
        self.assertEqual(len({r[0] for r in body}), d["invoices"])
        self.assertEqual(len({r[2] for r in body}), d["customers"])

    def test_readme_loader(self):
        readme = (DATA / "README.md").read_text(encoding="utf-8")
        blocks = re.findall(r"```python\n(.*?)```", readme, flags=re.DOTALL)
        loader = next(b for b in blocks if "def load_purchases" in b)
        namespace: dict = {}

        def no_network(*args, **kwargs):
            raise OSError("network disabled in tests")

        with (
            mock.patch.dict(os.environ, {"NLP_LLMS_DATA": str(DATA)}),
            mock.patch("urllib.request.urlopen", no_network),
        ):
            exec(compile(blocks[0] + "\n" + loader, "data/README.md", "exec"), namespace)  # noqa: S102
            rows = namespace["load_purchases"]()
        self.assertEqual(len(rows), DATASETS["purchases"]["lines"])
        self.assertEqual(
            hashlib.sha256((DATA / "purchases_v1.csv.gz").read_bytes()).hexdigest(),
            DATASETS["purchases"]["sha256"],
        )


@unittest.skipUnless(HAVE_NUMPY, "numpy is not installed")
class ProteinReference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        module = load_module(APPS / "protein" / "python" / "build.py", "protein_build")
        cls.data = module.analyze((DATA / "1ubq.pdb").read_text(encoding="ascii"))

    def test_recorded_numbers(self):
        meta = self.data["meta"]
        for key, value in PROTEIN.items():
            self.assertEqual(meta[key], value, key)
        self.assertTrue(meta["sequence"].startswith("MQIFVKTLTG"))
        self.assertEqual(len(self.data["contacts"]), PROTEIN["n_contacts"])
        self.assertTrue(all(j - i >= 3 for i, j in self.data["contacts"]))

    def test_writes_the_fields_the_page_reads(self):
        assert_fields(self, self.data, required_fields("protein"))

    def test_contacts_per_residue_add_up(self):
        total = sum(r["contacts"] for r in self.data["residues"])
        self.assertEqual(total, 2 * PROTEIN["n_contacts"])


@unittest.skipUnless(HAVE_RFM_STACK, "numpy, pandas or scikit-learn is not installed")
class RfmReference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        module = load_module(APPS / "rfm" / "python" / "build.py", "rfm_build")
        purchases, _ = module.load_purchases(str(DATA / "purchases_v1.csv.gz"))
        cls.data = module.build(purchases)
        cls.again = module.build(purchases)

    def test_deterministic(self):
        self.assertEqual(self.data, self.again)

    def test_segments(self):
        self.assertEqual(self.data["meta"]["k"], RFM_K)
        counts = {s["name"]: s["count"] for s in self.data["segments"]}
        self.assertEqual(set(counts), set(RFM_SEGMENTS))
        for name, n in RFM_SEGMENTS.items():
            self.assertLessEqual(abs(counts[name] - n), RFM_TOLERANCE, name)
        self.assertEqual(sum(counts.values()), DATASETS["purchases"]["customers"])

    def test_writes_the_fields_the_page_reads(self):
        assert_fields(self, self.data, required_fields("rfm"))


@unittest.skipUnless(rscript_ready(), "Rscript with bio3d, jsonlite, dplyr and cluster not found")
class RReferences(unittest.TestCase):
    def run_r(self, app: str, data_file: str) -> dict:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "data.json"
            script = APPS / app / "r" / "build.R"
            subprocess.run(
                ["Rscript", str(script), str(DATA / data_file), f"--out={out}"],
                check=True,
                capture_output=True,
            )
            return json.loads(out.read_text())

    def test_protein(self):
        data = self.run_r("protein", "1ubq.pdb")
        for key, value in PROTEIN.items():
            self.assertEqual(data["meta"][key], value, key)
        assert_fields(self, data, required_fields("protein"))

    def test_rfm(self):
        data = self.run_r("rfm", "purchases_v1.csv.gz")
        self.assertEqual(data["meta"]["k"], RFM_K)
        counts = {s["name"]: s["count"] for s in data["segments"]}
        for name, n in RFM_SEGMENTS.items():
            self.assertLessEqual(abs(counts[name] - n), RFM_TOLERANCE, name)
        assert_fields(self, data, required_fields("rfm"))


if __name__ == "__main__":
    unittest.main()
