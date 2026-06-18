# 00_build.R — Raw CSVs -> processed parquet (the layer 2->1 reproducibility bridge).
#
# This is the ONLY script that touches the raw Kaggle download. Run it once after
# python/fetch_data.py. Everything downstream (p1/p2/p3, report, app) reads the
# committed parquet in data/processed/ and never needs Kaggle or the 1 GB CSVs.
#
# Convention: arrow for fast CSV/parquet IO; dplyr for the data treatment.

library(dplyr)
library(arrow)

raw <- file.path("data", "raw")

tx  <- read_csv_arrow(file.path(raw, "train_transaction.csv"))
idn <- read_csv_arrow(file.path(raw, "train_identity.csv"))

# transactions : identity = 1 : (0 or 1). left_join keeps all 590,540 rows;
# identity columns are NA where the transaction was never fingerprinted (~76%).
# The very presence/absence of identity is signal.
model_input <- tx |>
  left_join(idn, by = "TransactionID")
stopifnot(nrow(model_input) == nrow(tx))

# Feature selection / cleaning is deferred to p2, which reads this full join and does
#   its own column treatment (keep TransactionID, isFraud, TransactionAmt, ProductCD,
#   time, and the chosen feature set; drop the rest). This build step just persists the
#   raw join; the committed *light* artifact (scored.parquet) is produced later by p2.

dir.create(file.path("data", "processed"), showWarnings = FALSE, recursive = TRUE)
write_parquet(model_input, file.path("data", "processed", "model_input.parquet"))
