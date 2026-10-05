# Customer segments from purchases: RFM, k-means, and data.json for a 3D scatter.
#
# Reference solution for Module 0, app 2, in R (dplyr + base kmeans(), silhouette from
# the cluster package that ships with R). Writes data.json next to this script for
# index.html to render with three.js.
#
#   Rscript agents-intro/rfm/r/build.R
#   Rscript agents-intro/rfm/r/build.R data/purchases_v1.csv.gz
#   Rscript agents-intro/rfm/r/build.R --out=/tmp/data.json       # write elsewhere
#   python -m http.server -d agents-intro/rfm/r 8000          # then open :8000
#
# The same steps and rules as the Python reference (agents-intro/rfm/python/build.py):
# 1. RFM as of 2026-01-01: recency = days since the last invoice, frequency = distinct
#    invoices, monetary = sum of quantity * unit_price.
# 2. log1p, then z-scores with the population SD (not R's sd(), which divides by n - 1).
# 3. k-means (20 random starts, set.seed(0)) for k = 3..6; keep the highest mean
#    silhouette. R's kmeans() and scikit-learn start differently, so segment sizes can
#    differ slightly between the two references.
# 4. Name segments from standardized centroids: recent = zR < 0, value = (zF + zM) / 2;
#    recent & value > 0.75 Champions, recent & value > 0 Loyal, recent New or promising,
#    not recent & value > 0 At risk, else Lost; repeated names get " (2)".

suppressPackageStartupMessages({
  library(dplyr)
  library(jsonlite)
  library(cluster)
})

args <- commandArgs(trailingOnly = FALSE)
script <- sub("^--file=", "", args[grep("^--file=", args)])
here <- if (length(script)) dirname(normalizePath(script)) else getwd()
trailing <- commandArgs(trailingOnly = TRUE)
out_arg <- sub("^--out=", "", grep("^--out=", trailing, value = TRUE))
local_file <- grep("^--out=", trailing, value = TRUE, invert = TRUE)[1]
repo_copy <- file.path(here, "..", "..", "..", "data", "purchases_v1.csv.gz")
urls <- c(
  "https://raw.githubusercontent.com/project-delphi/nlp-llms/main/data/purchases_v1.csv.gz",
  "https://cdn.jsdelivr.net/gh/project-delphi/nlp-llms@main/data/purchases_v1.csv.gz"
)
snapshot <- as.Date("2026-01-01")
k_range <- 3:6
colors <- c("#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#b07aa1", "#edc948")

read_purchases <- function(path) read.csv(gzfile(path), stringsAsFactors = FALSE)

load_purchases <- function() {
  if (!is.na(local_file)) return(list(df = read_purchases(local_file), source = basename(local_file)))
  if (file.exists(repo_copy)) {
    return(list(df = read_purchases(repo_copy), source = "data/purchases_v1.csv.gz"))
  }
  for (url in urls) {
    path <- tempfile(fileext = ".csv.gz")
    ok <- tryCatch(download.file(url, path, quiet = TRUE, mode = "wb") == 0,
                   error = function(e) FALSE, warning = function(w) FALSE)
    if (ok) return(list(df = read_purchases(path), source = url))
    message("Could not fetch ", url)
  }
  stop("Could not load the purchases table")
}

loaded <- load_purchases()
purchases <- loaded$df

# 1. RFM per customer.
rfm <- purchases |>
  mutate(invoice_date = as.Date(invoice_date), amount = quantity * unit_price) |>
  group_by(customer_id) |>
  summarise(
    recency = as.integer(snapshot - max(invoice_date)),
    frequency = n_distinct(invoice_id),
    monetary = sum(amount),
    .groups = "drop"
  ) |>
  arrange(customer_id)

# 2. log1p, then standardize with the population SD.
logged <- log1p(as.matrix(rfm[, c("recency", "frequency", "monetary")]))
log_mean <- colMeans(logged)
log_sd <- sqrt(colMeans(sweep(logged, 2, log_mean)^2))
z <- sweep(sweep(logged, 2, log_mean), 2, log_sd, "/")       # customers x 3

# 3. Choose k by mean silhouette.
fit <- function(k) { set.seed(0); kmeans(z, centers = k, nstart = 20, iter.max = 100) }
distances <- dist(z)
scores <- sapply(k_range, function(k) mean(silhouette(fit(k)$cluster, distances)[, "sil_width"]))
names(scores) <- k_range
k <- k_range[which.max(scores)]
model <- fit(k)

# Order segments by value (zF + zM), highest first.
value <- model$centers[, 2] + model$centers[, 3]
ord <- order(-value)
centers <- model$centers[ord, , drop = FALSE]
segment <- match(model$cluster, ord) - 1L                     # 0-based ids

# 4. Names from centroids.
name_of <- function(c) {
  v <- (c[2] + c[3]) / 2
  if (c[1] < 0) {
    if (v > 0.75) "Champions" else if (v > 0) "Loyal" else "New or promising"
  } else {
    if (v > 0) "At risk" else "Lost"
  }
}
base_names <- apply(centers, 1, name_of)
seg_names <- vapply(seq_along(base_names), function(s) {
  repeats <- sum(base_names[seq_len(s - 1)] == base_names[s])
  if (repeats) sprintf("%s (%d)", base_names[s], repeats + 1) else base_names[s]
}, character(1))
raw_centers <- expm1(sweep(sweep(centers, 2, log_sd, "*"), 2, log_mean, "+"))

segments <- lapply(seq_len(k), function(s) list(
  id = s - 1L,
  name = seg_names[s],
  color = colors[s],
  count = sum(segment == s - 1L),
  centroid = list(
    recency = round(raw_centers[s, 1], 1),
    frequency = round(raw_centers[s, 2], 2),
    monetary = round(raw_centers[s, 3], 2)
  ),
  centroid_z = round(unname(centers[s, ]), 4)
))
customers <- lapply(seq_len(nrow(rfm)), function(i) list(
  id = rfm$customer_id[i],
  recency = rfm$recency[i],
  frequency = rfm$frequency[i],
  monetary = round(rfm$monetary[i], 2),
  z = round(unname(z[i, ]), 4),
  segment = segment[i]
))
meta <- list(
  snapshot = format(snapshot),
  n_customers = nrow(rfm),
  n_invoices = n_distinct(purchases$invoice_id),
  k = k,
  k_rule = "highest mean silhouette over k = 3..6 (k-means, 20 restarts, seed 0)",
  silhouette = as.list(round(scores, 4)),
  transform = "z = (log1p(x) - mean) / sd, population sd",
  log_mean = round(unname(log_mean), 6),
  log_sd = round(unname(log_sd), 6),
  made_by = "r",
  source = loaded$source
)

out <- if (length(out_arg)) out_arg else file.path(here, "data.json")
write_json(list(meta = meta, segments = segments, customers = customers), out,
           auto_unbox = TRUE, digits = NA)
cat(sprintf("Read %s: %d lines, %d customers\n", loaded$source, nrow(purchases), nrow(rfm)))
cat(sprintf("Silhouette by k: %s -> k = %d\n",
            paste(sprintf("%d: %.4f", k_range, scores), collapse = ", "), k))
for (s in segments) {
  cat(sprintf("  %-18s %5d customers   centroid: %6.1f days, %5.2f invoices, %8.2f spend\n",
              s$name, s$count, s$centroid$recency, s$centroid$frequency, s$centroid$monetary))
}
cat(sprintf("Wrote %s\n", out))
