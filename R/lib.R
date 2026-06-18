# lib.R — Shared helpers used by p2/p3, the Quarto report, and the Shiny app.
# Single source of truth so the cost logic is never re-implemented per surface.
# Pure base-vector ops — works on any data.frame/tibble; no heavy deps.

# Base-case anchor for the cost ratio k = (fraud cost) / (friction cost).
# k=10 reads as "blocking a good customer costs 1/10th of eating a fraud": the
# headline P3 figure is reported here, and the sensitivity sweep (5:1 .. 50:1)
# brackets it. Single source so every surface agrees.
K_BASE <- 10

# Expected total cost of a block/approve policy, from the cost matrix.
# Point of view: Vesta (guaranteed payments) bears the chargeback, so BOTH error
# types fall in one coherent P&L.
#
# Cost model:
#   - False negative (approve a fraud): loss scales with the transaction value
#       cost_fn(t) = amount[t]            # value + merchandise + chargeback fee ~ amount
#   - False positive (block a good one): friction cost, taken as a fraction of the
#     same value, with k = (fraud cost) / (friction cost):
#       cost_fp(t) = amount[t] / k
#   k is the single free parameter the whole sensitivity analysis sweeps.
#
# scored:    data.frame/tibble with columns  amount, label (0/1), and the
#            calibrated PD named by `pred` (default "pd" = xgboost; "pd_logit" =
#            the Lasso baseline). Parameterizing the column lets every surface draw
#            either model's curve from the same cost logic — no re-implementation.
# threshold: block when PD >= threshold
expected_cost <- function(scored, threshold, k, pred = "pd") {
  blocked  <- scored[[pred]] >= threshold
  is_fraud <- scored$label == 1
  fn_cost <- sum(scored$amount[!blocked &  is_fraud])   # approved fraud -> fraud loss
  fp_cost <- sum(scored$amount[ blocked & !is_fraud]) / k  # blocked good -> friction
  fn_cost + fp_cost
}

# k-free decomposition of the cost curve. For each candidate block threshold,
# the dollars of approved fraud (FN side) and of blocked-good (FP side). Neither
# depends on k: the threshold decides WHO is blocked, k only rescales the FP side.
# Compute this once; every k (the sweep, the Shiny slider) then recombines as
# fn_dollars + fp_dollars / k without touching the 118k rows again.
cost_components <- function(scored, thresholds = seq(0, 1, by = 0.005), pred = "pd") {
  is_fraud <- scored$label == 1
  amt      <- scored$amount
  pd       <- scored[[pred]]
  fn <- vapply(thresholds, function(t) sum(amt[pd <  t &  is_fraud]), numeric(1))
  fp <- vapply(thresholds, function(t) sum(amt[pd >= t & !is_fraud]), numeric(1))
  data.frame(threshold = thresholds, fn_dollars = fn, fp_dollars = fp)
}

# Sweep thresholds for a given k; return the cost curve (threshold, fn_dollars,
# fp_dollars, cost). cost = fn + fp/k is exactly expected_cost() vectorized.
cost_curve <- function(scored, k, thresholds = seq(0, 1, by = 0.005), pred = "pd") {
  cc <- cost_components(scored, thresholds, pred = pred)
  cc$cost <- cc$fn_dollars + cc$fp_dollars / k
  cc
}

# The policy at a given k: the cost-minimizing threshold and what it costs, plus
# the two naive baselines the headline is measured against —
#   approve-all : block nobody (threshold > max pd) -> cost = all fraud dollars
#   block-0.5   : the off-the-shelf 0.5 cutoff that ignores the cost asymmetry
# Savings are baseline_cost - optimal_cost. Accepts a precomputed `components`
# so the k-sweep reuses one decomposition (pass cost_components(scored) once).
optimal_policy <- function(scored, k, thresholds = seq(0, 1, by = 0.005),
                           components = NULL, pred = "pd") {
  cc <- if (is.null(components)) cost_components(scored, thresholds, pred = pred) else components
  cost <- cc$fn_dollars + cc$fp_dollars / k
  i <- which.min(cost)
  approve_all <- expected_cost(scored, threshold = 1.0, k = k, pred = pred)  # max pd < 1 -> blocks none
  block_half  <- expected_cost(scored, threshold = 0.5, k = k, pred = pred)
  list(
    k                   = k,
    threshold           = cc$threshold[i],
    cost                = cost[i],
    cost_approve_all    = approve_all,
    cost_block_half     = block_half,
    savings_vs_approve  = approve_all - cost[i],
    savings_vs_block    = block_half  - cost[i]
  )
}
