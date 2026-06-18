# p2_baseline.R — Interpretable contrast to the xgboost model: an L1 (Lasso)
# logistic regression. Same 4-way temporal split and same Platt calibration as
# p2_model.R, so the cost curves in P3 compare like-for-like. The point is NOT to
# beat xgboost — it is to show a sparse, readable model as the floor, and to make
# "missingness is signal" explicit in a GLM (glmnet cannot route NA the way
# xgboost does natively).
#
# NA strategy (missingness is signal):
#   numeric     : impute with the TRAIN median + add a `_na` 0/1 indicator, so the
#                 model separates the effect of the value from the effect of absence.
#   categorical : NA becomes its own level "__NA__" before one-hot, so absence gets
#                 its own coefficient. High-cardinality factors lumped to top-100
#                 (by TRAIN frequency) + "__other__".
# Everything (medians, levels, lumping) is learned on TRAIN only and applied
# forward to select/calib/test — no leakage.

library(dplyr)
library(arrow)
library(glmnet)
library(Matrix)
library(yardstick)

set.seed(42)

dt <- read_parquet(file.path("data", "processed", "model_input.parquet"))

# --- Prep (mirror p2: drop zero-variance only; missingness is kept) ---------
nuniq <- vapply(dt, function(x) length(unique(x[!is.na(x)])), integer(1))
dt <- dt |> select(-all_of(names(nuniq)[nuniq <= 1]))
dt <- dt |> mutate(across(where(is.logical), as.integer),
                   across(where(is.character), as.factor))

# --- 4-way temporal split (identical boundaries to p2_model.R) --------------
day <- dt$TransactionDT / 86400
i_train  <- which(day <= 120)
i_select <- which(day > 120 & day <= 135)
i_calib  <- which(day > 135 & day <= 141)
i_test   <- which(day > 141)
cat(sprintf("split  train %d  select %d  calib %d  test %d\n",
            length(i_train), length(i_select), length(i_calib), length(i_test)))

y    <- dt$isFraud
feat <- dt |> select(-TransactionID, -isFraud, -TransactionDT)

is_fac <- vapply(feat, is.factor, logical(1))
num_cols <- names(feat)[!is_fac]
fac_cols <- names(feat)[ is_fac]
cat(sprintf("features %d  (numeric %d, categorical %d)\n",
            ncol(feat), length(num_cols), length(fac_cols)))

# --- Numeric: TRAIN-median impute + missing indicator -----------------------
num <- feat[, num_cols, drop = FALSE]
train_med <- vapply(num, function(x) median(x[i_train], na.rm = TRUE), numeric(1))
train_med[is.na(train_med)] <- 0  # column all-NA on train -> neutral 0

num_imp <- num
flags   <- vector("list", length(num_cols)); names(flags) <- num_cols
for (cn in num_cols) {
  na <- is.na(num_imp[[cn]])
  num_imp[[cn]][na] <- train_med[[cn]]
  if (any(na[i_train])) flags[[cn]] <- as.integer(na)  # keep flag only if train has NAs
}
flags <- flags[!vapply(flags, is.null, logical(1))]
flag_df <- as.data.frame(flags)
if (ncol(flag_df)) names(flag_df) <- paste0(names(flags), "_na")
cat(sprintf("numeric missing-indicators kept: %d\n", ncol(flag_df)))

# --- Categorical: lump to top-100 by TRAIN freq, with a FREQUENCY FLOOR ------
# The top-k cap bounds cardinality, but on a long-tailed factor the 100th level can
# still be a near-singleton (id_33 had levels with 5-8 train rows). A one-hot column
# for such a rare level is almost a row indicator: under L1 with standardize=TRUE its
# tiny SD makes the penalty on its original-scale coefficient ~lambda*SD ~ 0, so it
# can take an extreme value that perfectly separates train and does not generalize
# (the overconfident clump that degenerated the test PR curve). The floor (min_count)
# requires a level to clear a minimum TRAIN frequency before it earns its own column;
# everything below joins "__other__". 200/410601 ~= 0.05% — low enough to keep genuine
# signal, high enough to drop the near-separating singletons.
lump_na <- function(f, train_idx, k = 100, min_count = 200) {
  ch   <- as.character(f)
  freq <- sort(table(ch[train_idx]), decreasing = TRUE)
  top  <- names(freq)[seq_len(min(k, length(freq)))]
  keep <- top[freq[top] >= min_count]            # frequency floor: drop rare levels
  ch[!is.na(ch) & !(ch %in% keep)] <- "__other__"
  ch[is.na(ch)] <- "__NA__"
  factor(ch)  # levels fixed across all rows -> stable model.matrix columns
}
fac <- feat[, fac_cols, drop = FALSE]
for (cn in fac_cols) fac[[cn]] <- lump_na(fac[[cn]], i_train)

# --- Assemble sparse design matrix (one-hot factors + numerics + flags) -----
design <- cbind(num_imp, flag_df, fac)
X <- sparse.model.matrix(~ . - 1, data = design)
cat(sprintf("design matrix: %d rows x %d cols (sparse)\n", nrow(X), ncol(X)))

Xtr <- X[i_train, ]; Xse <- X[i_select, ]; Xca <- X[i_calib, ]; Xte <- X[i_test, ]

# --- Fit Lasso path on train; pick lambda by SELECT ROC-AUC -----------------
# Honors the temporal split (no random CV that would mix time windows).
# NOTE: select by ROC-AUC, not PR-AUC. On the heavily-regularized end of the path
# predictions are near-constant, where yardstick's PR-AUC degenerates to ~0.5 and
# would spuriously "win" — picking the null model. ROC-AUC is 0.5 for the null and
# well-behaved throughout. PR-AUC is still REPORTED on test for the xgboost contrast.
# lambda.min.ratio = 0.002 (vs glmnet's n>p default of 1e-4). With n (410601) >> p (968)
# the L1 path wants a small lambda, so the SELECT ROC-AUC keeps climbing deep into the
# path; at 0.01 it was still rising at the floor (0.8559 at lambda~5e-4) — the optimum
# was pinned to the search boundary, not interior. 0.002 (floor ~1e-4) lets the path go
# ~2 octaves lower so the AUC can turn over at a genuine interior maximum, while still
# stopping above the near-unregularized dense tail where each fit is brutally slow
# (that tail loses on SELECT AUC anyway). Trade-off: a few minutes vs 30+ for the full path.
fit <- glmnet(Xtr, y[i_train], family = "binomial", alpha = 1, standardize = TRUE,
              lambda.min.ratio = 0.002)
p_sel  <- predict(fit, Xse, type = "response")            # rows x nlambda
truth_sel <- factor(y[i_select], levels = c("1", "0"))
auc_sel <- apply(p_sel, 2, function(p) roc_auc_vec(truth_sel, p))
bi     <- which.max(auc_sel)
lam    <- fit$lambda[bi]
cat(sprintf("\nBEST select ROC-AUC %.4f  at lambda %.5g  (nonzero coefs %d)\n",
            auc_sel[bi], lam, fit$df[bi]))
idx <- unique(round(seq(1, length(fit$lambda), length.out = 8)))
cat("lambda path (select):\n")
print(data.frame(lambda = signif(fit$lambda[idx], 3), df = fit$df[idx],
                 roc_select = round(auc_sel[idx], 4)), row.names = FALSE)

# --- Test discrimination (out-of-sample) ------------------------------------
score_test  <- as.numeric(predict(fit, Xte, type = "response", s = lam))
truth_test  <- factor(y[i_test], levels = c("1", "0"))
cat("\n=== TEST discrimination (days 141-183) — Lasso logistic ===\n")
cat(sprintf("ROC-AUC: %.4f\n", roc_auc_vec(truth_test, score_test)))
cat(sprintf("PR-AUC : %.4f\n", pr_auc_vec(truth_test, score_test)))

# --- Platt calibration (fit on clean calib slice, apply to test) ------------
score_calib <- as.numeric(predict(fit, Xca, type = "response", s = lam))
eps <- 1e-6
lz  <- function(s) qlogis(pmin(pmax(s, eps), 1 - eps))
platt    <- glm(label ~ lz(score), data = data.frame(label = y[i_calib], score = score_calib),
                family = binomial)
pd_logit <- as.numeric(predict(platt, newdata = data.frame(score = score_test), type = "response"))

brier   <- function(y, p) mean((p - y)^2)
logloss <- function(y, p) { p <- pmin(pmax(p, 1e-15), 1 - 1e-15); -mean(y * log(p) + (1 - y) * log(1 - p)) }
yt <- y[i_test]
cat("\n=== TEST calibration — Lasso logistic ===\n")
cat(sprintf("%-8s %-10s %-10s %-10s\n", "method", "Brier", "logloss", "mean_pred"))
for (m in c("raw", "platt")) {
  p <- if (m == "raw") score_test else pd_logit
  cat(sprintf("%-8s %-10.5f %-10.5f %-10.4f\n", m, brier(yt, p), logloss(yt, p), mean(p)))
}
cat(sprintf("base fraud rate (test): %.4f\n", mean(yt)))

# --- Interpretability: the nonzero coefficients (the readable contrast) -----
co <- coef(fit, s = lam)
co <- data.frame(feature = rownames(co), beta = as.numeric(co))
co <- co[co$beta != 0 & co$feature != "(Intercept)", ]
co <- co[order(-abs(co$beta)), ]
cat(sprintf("\n=== top 20 |coef| (of %d nonzero) ===\n", nrow(co)))
print(head(co, 20), row.names = FALSE)

# --- Add pd_logit to the committed artifact (same test TransactionIDs) ------
base_test <- data.frame(TransactionID = dt$TransactionID[i_test], pd_logit = pd_logit)
scored_path <- file.path("data", "processed", "scored.parquet")
# mmap = FALSE: otherwise read_parquet memory-maps the file and the subsequent
# write to the SAME path fails on Windows ("file with a user-mapped section open").
scored <- read_parquet(scored_path, mmap = FALSE)
scored <- scored |>
  select(-any_of("pd_logit")) |>      # idempotent re-runs
  left_join(base_test, by = "TransactionID")
stopifnot(!anyNA(scored$pd_logit))
write_parquet(scored, scored_path)
cat("\nsaved: scored.parquet (added pd_logit column)\n")
