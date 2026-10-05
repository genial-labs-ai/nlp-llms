# Ubiquitin (PDB 1UBQ) in 3D: compute the structure's properties and write data.json.
#
# Reference solution for Module 0, app 1, in R, with bio3d and jsonlite. Writes data.json
# next to this script for index.html to render with three.js.
#
#   Rscript agents-intro/protein/r/build.R                    # RCSB, then the repo copy
#   Rscript agents-intro/protein/r/build.R data/1ubq.pdb      # a local file
#   Rscript agents-intro/protein/r/build.R --out=/tmp/data.json   # write elsewhere
#   python -m http.server -d agents-intro/protein/r 8000      # then open :8000
#
# The same choices as the Python reference (agents-intro/protein/python/build.py):
# - Contacts: CA-CA distance <= 8.0 Angstrom and |i - j| >= 3. Neighbors i+1 (3.8 A)
#   and i+2 (at most about 7.3 A) are always that close whatever the fold, so they are
#   excluded.
# - Radius of gyration, unweighted, for the 76 CA atoms and for all 602 protein heavy
#   atoms (the file has no hydrogens; the 58 waters are excluded), plus a mass-weighted
#   heavy-atom value.
# - Kyte-Doolittle hydrophobicity per residue, no window. B-factor of the CA atom.
#
# Unlike the Python reference, this script does not check the file's SHA-256 (base R
# has no SHA-256); it checks the atom and residue counts instead.

suppressPackageStartupMessages({
  library(bio3d)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = FALSE)
script <- sub("^--file=", "", args[grep("^--file=", args)])
here <- if (length(script)) dirname(normalizePath(script)) else getwd()
trailing <- commandArgs(trailingOnly = TRUE)
out_arg <- sub("^--out=", "", grep("^--out=", trailing, value = TRUE))
local_file <- grep("^--out=", trailing, value = TRUE, invert = TRUE)[1]
repo_copy <- file.path(here, "..", "..", "..", "data", "1ubq.pdb")

urls <- c(
  "https://files.rcsb.org/download/1UBQ.pdb",
  "https://raw.githubusercontent.com/project-delphi/nlp-llms/main/data/1ubq.pdb"
)
contact_cutoff <- 8.0
min_seq_sep <- 3

kyte_doolittle <- c(
  ALA = 1.8, ARG = -4.5, ASN = -3.5, ASP = -3.5, CYS = 2.5, GLN = -3.5, GLU = -3.5,
  GLY = -0.4, HIS = -3.2, ILE = 4.5, LEU = 3.8, LYS = -3.9, MET = 1.9, PHE = 2.8,
  PRO = -1.6, SER = -0.8, THR = -0.7, TRP = -0.9, TYR = -1.3, VAL = 4.2
)
masses <- c(C = 12.011, N = 14.007, O = 15.999, S = 32.06)

load_pdb <- function() {
  if (!is.na(local_file)) {
    return(list(pdb = read.pdb(local_file), source = basename(local_file)))
  }
  for (url in urls) {
    path <- tempfile(fileext = ".pdb")
    ok <- tryCatch(
      download.file(url, path, quiet = TRUE, mode = "wb") == 0,
      error = function(e) FALSE, warning = function(w) FALSE
    )
    if (ok) return(list(pdb = read.pdb(path), source = url))
    message("Could not fetch ", url)
  }
  if (file.exists(repo_copy)) {
    return(list(pdb = read.pdb(repo_copy), source = "data/1ubq.pdb (copy in this clone)"))
  }
  stop("Could not load 1UBQ from RCSB, the repository or a local copy")
}

radius_of_gyration <- function(xyz, w = rep(1, nrow(xyz))) {
  center <- colSums(xyz * w) / sum(w)
  sqrt(sum(w * rowSums(sweep(xyz, 2, center)^2)) / sum(w))
}

loaded <- load_pdb()
pdb <- loaded$pdb
protein <- pdb$atom[pdb$atom$type == "ATOM", ]          # waters are HETATM
heavy <- protein[protein$elesy != "H", ]
ca <- protein[protein$elety == "CA", ]
stopifnot(nrow(ca) == 76, nrow(heavy) == 602)

heavy_xyz <- as.matrix(heavy[, c("x", "y", "z")])        # 602 x 3
ca_xyz <- as.matrix(ca[, c("x", "y", "z")])              # 76 x 3

# Contacts: upper triangle of the CA distance matrix, |i - j| >= min_seq_sep.
d <- as.matrix(dist(ca_xyz))                             # 76 x 76
keep <- d <= contact_cutoff & (col(d) - row(d)) >= min_seq_sep
pairs <- which(keep, arr.ind = TRUE)
pairs <- pairs[order(pairs[, 1], pairs[, 2]), , drop = FALSE]
per_residue <- tabulate(c(pairs[, 1], pairs[, 2]), nbins = nrow(ca))

residues <- data.frame(
  index = seq_len(nrow(ca)) - 1L,                        # 0-based, like the Python build
  res_seq = ca$resno,
  res_name = ca$resid,
  code = aa321(ca$resid),
  x = ca$x, y = ca$y, z = ca$z,
  hydrophobicity = unname(kyte_doolittle[ca$resid]),
  contacts = per_residue,
  bfactor = ca$b
)

meta <- list(
  pdb_id = "1UBQ",
  title = "Ubiquitin, 1.8 Angstrom X-ray structure (Vijay-Kumar, Bugg & Cook, 1987)",
  n_residues = nrow(ca),
  n_heavy_atoms = nrow(heavy),
  sequence = paste(residues$code, collapse = ""),
  contact_cutoff_angstrom = contact_cutoff,
  contact_min_seq_sep = min_seq_sep,
  n_contacts = nrow(pairs),
  rg_ca_angstrom = round(radius_of_gyration(ca_xyz), 3),
  rg_heavy_angstrom = round(radius_of_gyration(heavy_xyz), 3),
  rg_heavy_mass_weighted_angstrom = round(
    radius_of_gyration(heavy_xyz, unname(masses[heavy$elesy])), 3
  ),
  hydrophobicity_scale = "Kyte-Doolittle (1982)",
  bfactor_atom = "CA",
  made_by = "r",
  source = loaded$source
)

out <- if (length(out_arg)) out_arg else file.path(here, "data.json")
contacts <- unname(lapply(seq_len(nrow(pairs)), function(k) unname(pairs[k, ]) - 1L))
write_json(
  list(meta = meta, residues = residues, contacts = contacts),
  out, auto_unbox = TRUE, digits = NA, pretty = TRUE
)
cat(sprintf("Read %s\n", loaded$source))
cat(sprintf("%d residues, %d heavy atoms: %s...\n", meta$n_residues, meta$n_heavy_atoms,
            substr(meta$sequence, 1, 10)))
cat(sprintf("Radius of gyration: %.3f A (CA), %.3f A (heavy)\n",
            meta$rg_ca_angstrom, meta$rg_heavy_angstrom))
cat(sprintf("Contacts (CA-CA <= 8 A, |i-j| >= 3): %d\n", meta$n_contacts))
cat(sprintf("Wrote %s\n", out))
