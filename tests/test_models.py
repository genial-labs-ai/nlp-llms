"""Model IDs in notebooks must match _variables.yml.

Notebooks cannot read _variables.yml on Colab, so they repeat the IDs. These
checks fail when the two drift apart in either direction.
"""

import json
import re
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
MODELS = yaml.safe_load((ROOT / "_variables.yml").read_text(encoding="utf-8"))["models"]

# Which `models` keys each notebook must quote verbatim.
USES = {
    "06-pretraining-huggingface": ["encoder", "encoder_cpu", "causal_lm"],
    "07-finetuning-lora": ["instruct_base", "instruct_base_revision"],
    "08-llm-apis": ["openai", "anthropic", "fallback", "instruct_base_revision"],
    "09-preference-learning": ["causal_lm"],
    "10-rlhf": ["causal_lm"],
    "11-calibration": ["openai", "anthropic", "fallback"],
    "12-rlcd-jev": ["jev", "openai", "anthropic"],
}

# Commercial model IDs: any such literal in a notebook must be a pinned value.
COMMERCIAL_ID = re.compile(r"""["']((?:gpt|claude)-[A-Za-z0-9.\-]+)["']""")


def code(slug: str) -> str:
    nb = json.loads((ROOT / "notebooks" / f"{slug}.ipynb").read_text(encoding="utf-8"))
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")


class ModelIds(unittest.TestCase):
    def test_notebooks_quote_the_pinned_ids(self):
        for slug, keys in USES.items():
            source = code(slug)
            for key in keys:
                with self.subTest(notebook=slug, key=key):
                    self.assertIn(f'"{MODELS[key]}"', source)

    def test_no_unpinned_commercial_ids(self):
        pinned = {str(v) for v in MODELS.values()}
        for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
            for model_id in COMMERCIAL_ID.findall(code(path.stem)):
                with self.subTest(notebook=path.stem, model_id=model_id):
                    self.assertIn(model_id, pinned)


if __name__ == "__main__":
    unittest.main()
