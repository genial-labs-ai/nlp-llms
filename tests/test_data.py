"""Checks on the `datasets` section of _variables.yml, data/ and data/README.md.

Nothing here touches the network. The loading snippet in data/README.md is
executed against the committed copies under data/.
"""

import hashlib
import os
import re
import unittest
from pathlib import Path
from unittest import mock

import yaml

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
V = yaml.safe_load((ROOT / "_variables.yml").read_text(encoding="utf-8"))
D = V["datasets"]
README = (DATA / "README.md").read_text(encoding="utf-8")
ENTRIES = {k: v for k, v in D.items() if isinstance(v, dict)}
ENTRY_FIELDS = {"name", "modules", "urls", "sha256", "bytes", "license", "license_url"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_readme_snippet() -> dict:
    """Execute the first Python block of data/README.md with the network disabled."""
    code = re.search(r"```python\n(.*?)```", README, flags=re.DOTALL).group(1)
    namespace: dict = {}

    def no_network(*args, **kwargs):
        raise OSError("network disabled in tests")

    with (
        mock.patch.dict(os.environ, {"NLP_LLMS_DATA": str(DATA)}),
        mock.patch("urllib.request.urlopen", no_network),
    ):
        exec(compile(code, "data/README.md", "exec"), namespace)  # noqa: S102
        namespace["topics"] = namespace["load_topics"]()
        namespace["lm"] = namespace["load_lm_corpus"]()
    return namespace


class Schema(unittest.TestCase):
    def test_every_entry_has_all_fields(self):
        for key, d in ENTRIES.items():
            self.assertEqual(ENTRY_FIELDS - set(d), set(), key)
            self.assertRegex(d["sha256"], r"^[0-9a-f]{64}$", key)
            self.assertTrue(d["urls"], key)
            for url in d["urls"]:
                self.assertTrue(url.startswith("https://"), url)

    def test_modules_exist(self):
        numbers = {m["n"] for m in V["modules"].values()}
        for key, d in ENTRIES.items():
            self.assertLessEqual(set(d["modules"]), numbers, key)

    def test_fallback_base_matches_repo(self):
        r = V["repo"]
        self.assertEqual(
            D["fallback_base"],
            f"https://raw.githubusercontent.com/{r['owner']}/{r['name']}/{r['branch']}/data",
        )

    def test_committed_copies_are_listed_under_the_fallback_base(self):
        for key, d in ENTRIES.items():
            if "file" in d:
                self.assertIn(f"{D['fallback_base']}/{d['file']}", d["urls"], key)

    def test_check_date(self):
        self.assertRegex(str(D["checked"]), r"^\d{4}-\d{2}-\d{2}$")


class CommittedCopies(unittest.TestCase):
    def test_hash_and_size_match(self):
        for key, d in ENTRIES.items():
            if "file" not in d:
                continue
            path = DATA / d["file"]
            self.assertTrue(path.exists(), path)
            self.assertEqual(sha256(path), d["sha256"], key)
            self.assertEqual(path.stat().st_size, d["bytes"], key)

    def test_data_directory_stays_small(self):
        files = [p for p in DATA.rglob("*") if p.is_file() and ".cache" not in p.parts]
        total = sum(p.stat().st_size for p in files)
        # 6 MB until 2026-10-06; Lab 9's three files added 5.99 MB (11.40 MB in all).
        self.assertLess(total, 12_000_000)


class Readme(unittest.TestCase):
    def test_readme_agrees_with_variables(self):
        for key, d in ENTRIES.items():
            self.assertIn(d["sha256"], README, key)
            self.assertIn(d["license_url"], README, key)
            for url in d["urls"]:
                self.assertIn(url, README, key)
        self.assertIn(str(D["checked"]), README)

    def test_loading_snippet_gives_the_recorded_splits(self):
        ns = run_readme_snippet()
        topics, lm = ns["topics"], ns["lm"]
        self.assertEqual(ns["TOPIC_CLASSES"], D["topics"]["classes"])
        for split, n in D["topics"]["splits"].items():
            texts, labels = topics[split]
            self.assertEqual((len(texts), len(labels)), (n, n), split)
            per_class = n // len(D["topics"]["classes"])
            for label in range(len(D["topics"]["classes"])):
                self.assertEqual(labels.count(label), per_class, (split, label))
        for split, (start, end) in D["lm"]["splits"].items():
            self.assertEqual(len(lm[split]), end - start, split)
        self.assertEqual(sum(len(s) for s in lm.values()), D["lm"]["bytes"])

    def test_topic_splits_do_not_overlap(self):
        topics = run_readme_snippet()["topics"]
        seen: set = set()
        for texts, _ in topics.values():
            self.assertEqual(len(set(texts)), len(texts))
            self.assertEqual(seen & set(texts), set())
            seen |= set(texts)


if __name__ == "__main__":
    unittest.main()
