# p2_model.R — "Why the threshold is an economic decision": the model + calibration.
# The model is the means, not the headline. We need a CALIBRATED probability so the
# costs in p3 can actually be summed — an AUC-chaser never calibrates.
#
# Design (light, principled tuning — option b) with a 4-WAY TEMPORAL split so the
# calibration set is never seen by model selection (no leakage, no double-use):
#   train  1-120d  : fit the model
#   select 120-135d: early stopping + hyperparameter search
#   calib  135-141d: ONLY calibration (model never watched it) — closest window to test
#   test   141-183d: untouched final evaluation
#   - small random search over the few params that matter; selection by PR-AUC.
#   - NO class rebalancing (no scale_pos_weight): it distorts probabilities and we
#     care about calibration; natural prevalence is kept and calibrated downstream.

library(dplyr)
library(arrow)
library(xgboost)
library(yardstick)

set.seed(42)

dt <- read_parquet(file.path("data", "processed", "model_input.parquet"))

# --- Prep -------------------------------------------------------------------
# Drop zero-variance columns only. Missingness is signal here, so we do
# NOT drop high-NA columns.
nuniq <- vapply(dt, function(x) length(unique(x[!is.na(x)])), integer(1))
dt <- dt |> select(-all_of(names(nuniq)[nuniq <= 1]))

# logical -> int; character -> factor (xgboost native categorical via feature_types)
dt <- dt |> mutate(across(where(is.logical), as.integer),
                   across(where(is.character), as.factor))

# Lump very-high-cardinality factors to top-100 + __other__, preserving NA, so the
# native categorical handling does not choke on thousands of levels (DeviceInfo, ...).
lump_factor <- function(f, k = 100) {
  if (!is.factor(f) || nlevels(f) <= k) return(f)
  top <- names(sort(table(f), decreasing = TRUE))[seq_len(k)]
  ch <- as.character(f); ch[!is.na(ch) & !(ch %in% top)] <- "__other__"
  factor(ch)
}
dt <- dt |> mutate(across(where(is.factor), lump_factor))

# --- 4-way temporal split ---------------------------------------------------
day <- dt$TransactionDT / 86400
i_train  <- which(day <= 120)
i_select <- which(day > 120 & day <= 135)
i_calib  <- which(day > 135 & day <= 141)
i_test   <- which(day > 141)
cat(sprintf("split  train %d  select %d  calib %d  test %d\n",
            length(i_train), length(i_select), length(i_calib), length(i_test)))

y    <- dt$isFraud
# Raw TransactionDT left OUT of features (absolute time -> would not generalize).
feat <- dt |> select(-TransactionID, -isFraud, -TransactionDT) |> as.data.frame()
ftypes <- ifelse(vapply(feat, is.factor, logical(1)), "c", "q")
cat(sprintf("features %d  (categorical %d)\n", length(ftypes), sum(ftypes == "c")))

dtrain  <- xgb.DMatrix(feat[i_train, ],  label = y[i_train],  feature_types = ftypes)
dselect <- xgb.DMatrix(feat[i_select, ], label = y[i_select], feature_types = ftypes)
dcalib  <- xgb.DMatrix(feat[i_calib, ],  label = y[i_calib],  feature_types = ftypes)
dtest   <- xgb.DMatrix(feat[i_test, ],   label = y[i_test],   feature_types = ftypes)

# --- Light random search (selection by select PR-AUC) ----------------------
grid <- list(
  max_depth        = c(4, 5, 6, 7, 8),
  eta              = c(0.02, 0.03, 0.05, 0.08),
  min_child_weight = c(1, 3, 5, 10),
  reg_lambda       = c(0.5, 1, 2, 5),
  reg_alpha        = c(0, 0.5, 1),
  subsample        = c(0.7, 0.8, 0.9),
  colsample_bytree = c(0.6, 0.8, 1.0)
)
N_COMBOS <- 16
best <- list(score = -Inf)

for (i in seq_len(N_COMBOS)) {
  combo <- lapply(grid, function(v) sample(v, 1))
  params <- c(list(objective = "binary:logistic", eval_metric = "aucpr",
                   tree_method = "hist"), combo)
  t0 <- Sys.time()
  fit <- xgb.train(params, dtrain, nrounds = 800,
                   evals = list(select = dselect), early_stopping_rounds = 40, verbose = 0)
  score <- as.numeric(xgb.attr(fit, "best_score"))
  iter  <- as.integer(xgb.attr(fit, "best_iteration"))
  cat(sprintf("[%2d/%d] PR-AUC %.4f  iter %3d  depth %d eta %.2f mcw %2d lam %.1f alpha %.1f sub %.1f col %.1f  (%.0fs)\n",
              i, N_COMBOS, score, iter, combo$max_depth, combo$eta, combo$min_child_weight,
              combo$reg_lambda, combo$reg_alpha, combo$subsample, combo$colsample_bytree,
              as.numeric(difftime(Sys.time(), t0, units = "secs"))))
  if (score > best$score) best <- list(score = score, iter = iter, params = params, fit = fit)
}

cat(sprintf("\nBEST select PR-AUC %.4f  iter %d\n", best$score, best$iter))
str(best$params)
fit <- best$fit

# --- Discrimination on the untouched test window ----------------------------
score_test <- predict(fit, dtest)
ev <- tibble(truth = factor(y[i_test], levels = c("1", "0")), score = score_test)
cat(sprintf("\n=== TEST discrimination (days 141-183, out-of-sample) ===\n"))
cat(sprintf("ROC-AUC: %.4f\n", roc_auc_vec(ev$truth, ev$score)))
cat(sprintf("PR-AUC : %.4f\n", pr_auc_vec(ev$truth, ev$score)))

# --- Calibration: fit on the clean CALIB slice, evaluate on TEST ------------
score_calib <- predict(fit, dcalib)
cal_df <- tibble(label = y[i_calib], score = score_calib)
brier   <- function(y, p) mean((p - y)^2)
logloss <- function(y, p) { p <- pmin(pmax(p, 1e-15), 1 - 1e-15); -mean(y * log(p) + (1 - y) * log(1 - p)) }

# Platt: logistic on logit(score)
eps <- 1e-6
lz  <- function(s) qlogis(pmin(pmax(s, eps), 1 - eps))
platt    <- glm(label ~ lz(score), data = cal_df, family = binomial)
pd_platt <- as.numeric(predict(platt, newdata = data.frame(score = score_test), type = "response"))
# Isotonic: isoreg() returns yf in SORTED-x order, so pair with sort(score).
ir      <- isoreg(cal_df$score, cal_df$label)
iso_fun <- approxfun(sort(cal_df$score), ir$yf, method = "constant", rule = 2, ties = "ordered")
pd_iso  <- iso_fun(score_test)

yt <- y[i_test]
cat(sprintf("\n=== TEST calibration (fit on calib 135-141d) ===\n"))
cat(sprintf("base fraud rate (test): %.4f\n", mean(yt)))
cat(sprintf("%-8s %-10s %-10s %-10s\n", "method", "Brier", "logloss", "mean_pred"))
for (m in c("raw", "platt", "iso")) {
  p <- switch(m, raw = score_test, platt = pd_platt, iso = pd_iso)
  cat(sprintf("%-8s %-10.5f %-10.5f %-10.4f\n", m, brier(yt, p), logloss(yt, p), mean(p)))
}

rel <- function(p, y) {
  br <- unique(quantile(p, seq(0, 1, .1), na.rm = TRUE))
  tibble(bin = cut(p, br, include.lowest = TRUE), p = p, y = y) |>
    group_by(bin) |>
    summarise(n = n(), mean_pred = mean(p), obs_rate = mean(y), .groups = "drop")
}
cat("\n=== reliability raw (test) ===\n");   print(rel(score_test, yt), n = 12)
cat("\n=== reliability platt (test) ===\n"); print(rel(pd_platt,  yt), n = 12)
cat("\n=== reliability iso (test) ===\n");   print(rel(pd_iso,    yt), n = 12)

# --- Persist for the baseline / cost stages ---------------------------------
xgb.save(fit, file.path("data", "processed", "xgb_model.json"))
preds <- list(
  calib = cal_df,
  test  = dt[i_test, ] |>
    transmute(TransactionID, ProductCD, amount = TransactionAmt, label = isFraud) |>
    mutate(score = score_test, pd_platt = pd_platt, pd_iso = pd_iso)
)
saveRDS(preds, file.path("data", "processed", "p2_preds.rds"))
cat("\nsaved: xgb_model.json, p2_preds.rds\n")

# --- Emit the committed layer-1 artifact: scored.parquet --------------------
# The bridge P2 -> P3. Calibrator chosen = Platt (best logloss + reliability in
# the decision region; smoother tails than isotonic). One row per out-of-sample
# test transaction: the calibrated PD plus the minimum P3 needs to turn it into
# a dollar decision. This is the ONLY processed file that is committed.
scored <- preds$test |>
  transmute(TransactionID, ProductCD, amount, label, score, pd = pd_platt)
write_parquet(scored, file.path("data", "processed", "scored.parquet"))
cat("saved: scored.parquet\n")

# --- NEXT -------------------------------------------------------------------
# Baseline: glmnet Lasso logistic as an interpretable contrast (report-time).
