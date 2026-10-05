"""Ubiquitin (PDB 1UBQ) in 3D: compute the structure's properties and write data.json.

Reference solution for Module 0, app 1, in Python. Reads the PDB file with plain text
parsing and NumPy (no Biopython), then writes data.json next to this script for
index.html to render with three.js.

    python agents-intro/protein/python/build.py              # RCSB, then the repo copy
    python agents-intro/protein/python/build.py --pdb data/1ubq.pdb   # a local file
    python -m http.server -d agents-intro/protein/python 8000         # then open :8000

What it computes, and the choices made:

- C-alpha backbone: one CA atom per residue, 76 residues, in chain order.
- Contact map: residue pairs (i, j) whose CA atoms are at most 8.0 Angstrom apart, with
  |i - j| >= 3. Neighbors i+1 (3.8 Angstrom, the fixed CA-CA bond geometry) and i+2
  (at most about 7.3 Angstrom) are always within 8 Angstrom whatever the fold, so they
  say nothing about how the chain packs; they are excluded.
- Radius of gyration, unweighted, sqrt(mean |r - mean(r)|^2), for two atom sets:
  the 76 CA atoms, and all 602 protein heavy atoms (every ATOM record; the file has no
  hydrogens). The 58 waters (HETATM HOH) are excluded. A mass-weighted heavy-atom value
  is also recorded, because tools such as GROMACS weight by mass.
- Kyte-Doolittle hydrophobicity: the scale value of each residue's amino acid
  (Kyte & Doolittle 1982, J. Mol. Biol. 157:105-132). No window averaging.
- B-factor per residue: the CA atom's B-factor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "data.json"
REPO_COPY = HERE.parents[2] / "data" / "1ubq.pdb"  # present when run inside a clone

# The same entry as `datasets.ubiquitin` in _variables.yml (tests check they agree).
URLS = [
    "https://files.rcsb.org/download/1UBQ.pdb",
    "https://raw.githubusercontent.com/project-delphi/nlp-llms/main/data/1ubq.pdb",
]
SHA256 = "d4a6812d8951cf6594e6a0763f089e35f5a80b62acb3c117b2c5565228a7b161"

CONTACT_CUTOFF = 8.0  # Angstrom, between CA atoms
MIN_SEQ_SEP = 3  # pairs with |i - j| < 3 are excluded (see the module docstring)

KYTE_DOOLITTLE = {
    "ALA": 1.8, "ARG": -4.5, "ASN": -3.5, "ASP": -3.5, "CYS": 2.5,
    "GLN": -3.5, "GLU": -3.5, "GLY": -0.4, "HIS": -3.2, "ILE": 4.5,
    "LEU": 3.8, "LYS": -3.9, "MET": 1.9, "PHE": 2.8, "PRO": -1.6,
    "SER": -0.8, "THR": -0.7, "TRP": -0.9, "TYR": -1.3, "VAL": 4.2,
}  # fmt: skip
ONE_LETTER = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
    "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
    "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}  # fmt: skip
MASS = {"C": 12.011, "N": 14.007, "O": 15.999, "S": 32.06}


def load_pdb(path: str | None) -> tuple[str, str]:
    """Return (text, source). Each download is checked against SHA256."""
    if path:
        return Path(path).read_text(), Path(path).name
    for url in URLS:
        try:
            with urllib.request.urlopen(url, timeout=20) as response:
                blob = response.read()
        except OSError as error:
            print(f"Could not fetch {url}: {error}")
            continue
        if hashlib.sha256(blob).hexdigest() != SHA256:
            print(f"Ignoring {url}: it does not match the recorded file (a newer revision?)")
            continue
        return blob.decode("ascii"), url
    if REPO_COPY.exists():
        return REPO_COPY.read_text(), "data/1ubq.pdb (copy in this clone)"
    raise RuntimeError("Could not load 1UBQ from RCSB, the repository or a local copy")


def parse_atoms(text: str) -> list[dict]:
    """Protein atoms (ATOM records) of the first model, by PDB fixed columns."""
    atoms = []
    for line in text.splitlines():
        if line.startswith("ENDMDL"):
            break
        if not line.startswith("ATOM"):
            continue
        atoms.append(
            {
                "name": line[12:16].strip(),
                "alt": line[16].strip(),
                "res_name": line[17:20],
                "chain": line[21],
                "res_seq": int(line[22:26]),
                "xyz": (float(line[30:38]), float(line[38:46]), float(line[46:54])),
                "b": float(line[60:66]),
                "element": line[76:78].strip() or line[12:14].strip(),
            }
        )
    # Keep the first alternate location only (1UBQ has none).
    return [a for a in atoms if a["alt"] in ("", "A")]


def radius_of_gyration(xyz: np.ndarray, weights: np.ndarray | None = None) -> float:
    """sqrt(sum w |r - r_mean|^2 / sum w); unweighted when weights is None."""
    w = np.ones(len(xyz)) if weights is None else weights
    center = (w[:, None] * xyz).sum(0) / w.sum()  # (3,)
    return float(np.sqrt((w * ((xyz - center) ** 2).sum(1)).sum() / w.sum()))


def contacts(ca: np.ndarray, cutoff: float, min_sep: int) -> list[tuple[int, int]]:
    """Index pairs (i, j), i < j, with CA-CA distance <= cutoff and j - i >= min_sep."""
    dist = np.linalg.norm(ca[:, None, :] - ca[None, :, :], axis=-1)  # (n, n)
    i, j = np.nonzero(np.triu(dist <= cutoff, k=min_sep))
    return list(zip(i.tolist(), j.tolist(), strict=True))


def analyze(text: str) -> dict:
    atoms = parse_atoms(text)
    heavy = [a for a in atoms if a["element"] != "H"]
    ca_atoms = [a for a in atoms if a["name"] == "CA"]
    heavy_xyz = np.array([a["xyz"] for a in heavy])  # (602, 3)
    ca = np.array([a["xyz"] for a in ca_atoms])  # (76, 3)
    masses = np.array([MASS[a["element"]] for a in heavy])

    pairs = contacts(ca, CONTACT_CUTOFF, MIN_SEQ_SEP)
    per_residue = np.zeros(len(ca), dtype=int)
    for i, j in pairs:
        per_residue[i] += 1
        per_residue[j] += 1

    residues = []
    for k, a in enumerate(ca_atoms):
        residues.append(
            {
                "index": k,
                "res_seq": a["res_seq"],
                "res_name": a["res_name"],
                "code": ONE_LETTER[a["res_name"]],
                "x": round(a["xyz"][0], 3),
                "y": round(a["xyz"][1], 3),
                "z": round(a["xyz"][2], 3),
                "hydrophobicity": KYTE_DOOLITTLE[a["res_name"]],
                "contacts": int(per_residue[k]),
                "bfactor": a["b"],
            }
        )

    meta = {
        "pdb_id": "1UBQ",
        "title": "Ubiquitin, 1.8 Angstrom X-ray structure (Vijay-Kumar, Bugg & Cook, 1987)",
        "n_residues": len(ca),
        "n_heavy_atoms": len(heavy),
        "sequence": "".join(r["code"] for r in residues),
        "contact_cutoff_angstrom": CONTACT_CUTOFF,
        "contact_min_seq_sep": MIN_SEQ_SEP,
        "n_contacts": len(pairs),
        "rg_ca_angstrom": round(radius_of_gyration(ca), 3),
        "rg_heavy_angstrom": round(radius_of_gyration(heavy_xyz), 3),
        "rg_heavy_mass_weighted_angstrom": round(radius_of_gyration(heavy_xyz, masses), 3),
        "hydrophobicity_scale": "Kyte-Doolittle (1982)",
        "bfactor_atom": "CA",
        "made_by": "python",
    }
    return {"meta": meta, "residues": residues, "contacts": [list(p) for p in pairs]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute 1UBQ properties for index.html.")
    parser.add_argument("--pdb", help="read this PDB file instead of downloading")
    parser.add_argument("--out", default=str(OUT), help="where to write data.json")
    args = parser.parse_args()

    text, source = load_pdb(args.pdb)
    data = analyze(text)
    data["meta"]["source"] = source
    Path(args.out).write_text(json.dumps(data, indent=1) + "\n")
    m = data["meta"]
    print(f"Read {source}")
    print(f"{m['n_residues']} residues, {m['n_heavy_atoms']} heavy atoms: {m['sequence'][:10]}...")
    print(f"Radius of gyration: {m['rg_ca_angstrom']} A (CA), {m['rg_heavy_angstrom']} A (heavy)")
    print(f"Contacts (CA-CA <= 8 A, |i-j| >= 3): {m['n_contacts']}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
