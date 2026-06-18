# p3_policy.R — "Where to put the threshold": the policy with a dollar figure.
# Consumes the calibrated PD from p2 (scored.parquet) and the cost matrix from lib.R.
#
# Two views:
#   (1) base case at k = K_BASE: the cost-minimizing threshold and what it saves
#       over the naive baselines (approve-all, block-at-0.5);
#   (2) sensitivity: how that optimal threshold and the savings move as k sweeps
#       5:1 .. 50:1 — the consultant move that shields against "you guessed the
#       numbers". The Shiny app makes this interactive.

library(arrow)
source(file.path("R", "lib.R"))

scored <- read_parquet(file.path("data", "processed", "scored.parquet"))

usd <- function(x) paste0("$", format(round(x), big.mark = ",", scientific = FALSE))

# All figures are on the P2 test window (out-of-sample, days 141-183), not monthly.
cat(sprintf("Test window: %s transactions, %.2f%% fraud, %s in fraud dollars.\n\n",
            format(nrow(scored), big.mark = ","),
            100 * mean(scored$label == 1),
            usd(sum(scored$amount[scored$label == 1]))))

# Compute the k-free decomposition once; every k below reuses it (no recompute).
comp <- cost_components(scored)

# --- (1) Base case: optimal threshold at the anchor k ----------------------
base <- optimal_policy(scored, K_BASE, components = comp)
cat(sprintf("=== Base case  (k = %d) ===\n", K_BASE))
cat(sprintf("  Optimal block threshold : %.3f\n", base$threshold))
cat(sprintf("  Total cost (optimal)    : %s\n", usd(base$cost)))
cat(sprintf("  vs approve-all          : %s  (saves %s)\n",
            usd(base$cost_approve_all), usd(base$savings_vs_approve)))
cat(sprintf("  vs block-at-0.5         : %s  (saves %s)\n\n",
            usd(base$cost_block_half), usd(base$savings_vs_block)))

# --- (2) Sensitivity: how the optimal threshold moves with k ---------------
# A good model's optimal threshold should slide DOWN as fraud gets more costly
# (k up) — block more aggressively. A flat line would mean the policy ignores k.
ks <- c(5, 10, 15, 20, 30, 50)
sens <- do.call(rbind, lapply(ks, function(k) {
  p <- optimal_policy(scored, k, components = comp)
  data.frame(k = p$k, threshold = p$threshold, cost = p$cost,
             savings_vs_block = p$savings_vs_block,
             savings_vs_approve = p$savings_vs_approve)
}))

cat("=== Sensitivity to the cost ratio k ===\n")
print(transform(sens,
                cost               = usd(cost),
                savings_vs_block   = usd(savings_vs_block),
                savings_vs_approve = usd(savings_vs_approve)),
      row.names = FALSE)
