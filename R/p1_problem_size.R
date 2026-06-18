# P1 — Problem size: how much is at stake, and where is the loss concentrated?
# Pre-model, descriptive. Uses train_transaction only (P1 is about money, not the model).

library(data.table)

tx <- fread(file.path("data", "raw", "train_transaction.csv"),
            select = c("TransactionID", "isFraud", "TransactionAmt", "ProductCD"))

# --- 1. Headline figures: exposure in count vs. in US$ ---------------------
headline <- tx[, .(
  n_tx           = .N,
  fraud_rate_cnt = mean(isFraud),
  total_usd      = sum(TransactionAmt),
  fraud_usd      = sum(TransactionAmt[isFraud == 1]),
  fraud_rate_usd = sum(TransactionAmt[isFraud == 1]) / sum(TransactionAmt),
  ticket_legit   = mean(TransactionAmt[isFraud == 0]),
  ticket_fraud   = mean(TransactionAmt[isFraud == 1])
)]
cat("\n===== P1 HEADLINE =====\n")
print(t(headline))

# --- 2. Concentrated or spread? Fraud US$ by value band --------------------
tx[, amt_band := cut(TransactionAmt,
       breaks = c(-Inf, 50, 100, 250, 500, 1000, Inf),
       labels = c("<=50","50-100","100-250","250-500","500-1000",">1000"))]

by_band <- tx[, .(
  n_tx        = .N,
  fraud_usd   = sum(TransactionAmt[isFraud == 1])
), by = amt_band][order(amt_band)]
by_band[, fraud_usd_share := fraud_usd / sum(fraud_usd)]
cat("\n===== FRAUD US$ BY VALUE BAND =====\n")
print(by_band)

# Share of fraud US$ coming from tickets above the median (69)
hi <- tx[TransactionAmt > 69, sum(TransactionAmt[isFraud == 1])] /
      tx[, sum(TransactionAmt[isFraud == 1])]
cat(sprintf("\nFraud US$ from tickets > median(69): %.1f%%\n", 100 * hi))

# --- 3. Texture: fraud by product (risk is not uniform) --------------------
by_prod <- tx[, .(
  n_tx           = .N,
  fraud_rate_cnt = mean(isFraud),
  fraud_usd      = sum(TransactionAmt[isFraud == 1])
), by = ProductCD][order(-fraud_usd)]
by_prod[, fraud_usd_share := fraud_usd / sum(fraud_usd)]
cat("\n===== FRAUD BY PRODUCT =====\n")
print(by_prod)

# --- 4. Persist a light summary for the report ------------------------------
# The report (reproducibility layer 1) renders from committed artifacts with no
# Kaggle: P2/P3 recompute from scored.parquet, but P1 reads the raw CSV (gitignored),
# so emit these three tiny tables as the committed P1 surface. Mirrors how p2/p3
# emit their artifacts. Regenerating it is layer 2 (needs the raw via fetch_data.py).
p1_summary <- list(
  headline    = headline,
  by_band     = by_band,
  by_product  = by_prod,
  hi_median   = hi,            # share of fraud US$ above the median ticket
  median_amt  = 69            # TransactionAmt median
)
saveRDS(p1_summary, file.path("data", "processed", "p1_summary.rds"))
cat("\nsaved: p1_summary.rds\n")
