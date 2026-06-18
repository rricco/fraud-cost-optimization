# Fraud Cost Optimization

**Where to set a payment-fraud block threshold to minimize total cost — fraud loss vs. customer friction.**

Across a ~6-month book of **590,540** card-not-present transactions worth **$79.7M**, fraud
accounts for **$3.08M** of exposure. The usual reflex is to chase a better classifier (AUC).
But a model only *ranks* transactions; someone still has to decide *at what probability we
actually block one* — and that cutoff trades fraud loss against the cost of declining good
customers. On the held-out test window, choosing that threshold by **expected dollar cost**
instead of the off-the-shelf 0.5 rule saves **$115k** with the *same model* (just the right
cut), and **$287k** versus approving everything. **The threshold is a risk decision, not a
leaderboard score.**

📄 **[Read the full report](report/report.html)** · 🖱️ Interactive threshold/savings explorer in [`app/app.R`](app/app.R)

---

## The argument, in three parts

**P1 — How much is at stake.** Fraud is **3.5% of transactions by count** but **3.9% by
dollars**, because the average fraudulent ticket ($149) runs larger than the average
legitimate one ($135). The loss is concentrated in *value, not volume*: **88%** of fraud
dollars come from tickets above the median ($69). The policy should be value-weighted —
expected loss is `PD × exposure`, not a flat rate.

**P2 — Why the threshold is economic.** A gradient-boosted model (xgboost) ranks fraud well
(test **ROC-AUC 0.90 / PR-AUC 0.53**), with an interpretable L1-logistic (Lasso) baseline as
the floor (**0.82 / 0.20**). But ranking is necessary, not sufficient: to *sum dollars* the
score must be a **calibrated probability**, so it is Platt-scaled on a held-out window. A
4-way temporal split (train / select / calibrate / test) keeps the calibration set unseen by
model selection — no leakage.

**P3 — Where to put the threshold.** With one free parameter **k = (fraud cost) / (friction
cost)**, total expected cost is `fn_dollars(t) + fp_dollars(t) / k`. At the base case k = 10
the cost-minimizing block threshold is **0.105** — far below 0.5 — and the recommendation is
**robust**: the 0.5 cutoff never wins at any k, and savings versus approving everything stay
between **$221k (k=5)** and **$440k (k=50)** across the whole range. You don't have to nail k
exactly.

> All P2/P3 figures are recomputed live from the committed `scored.parquet`; P1 reads a
> committed summary of the raw table. Numbers are on the out-of-sample test window
> (days 141–183), not an annual run-rate.

## Repository layout

```
R/
  00_build.R          raw CSVs -> processed parquet (the only script that touches Kaggle)
  lib.R               shared cost model + policy helpers (one source of truth)
  p1_problem_size.R   P1: exposure in dollars vs. count
  p2_model.R          P2: xgboost + Platt calibration, 4-way temporal split
  p2_baseline.R       P2: interpretable L1 (Lasso) logistic baseline
  p3_policy.R         P3: cost-minimizing threshold + sensitivity to k
app/app.R             Shiny app: drag k, watch the threshold and savings move
report/report.qmd     Quarto report (rendered: report/report.html)
python/fetch_data.py  fetch the IEEE-CIS dataset from Kaggle
data/processed/       committed light artifacts (scored.parquet, p1_summary.rds)
```

## Reproduce

The analysis is reproducible **without Kaggle**: the committed `data/processed/` artifacts let
you re-render the report and run the app directly.

```r
# dependencies are pinned with renv
renv::restore()

# re-render the report (P1/P2/P3 figures recompute from the committed parquet)
quarto::quarto_render("report/report.qmd")

# launch the interactive threshold explorer
shiny::runApp("app")
```

To rebuild from raw data: run `python/fetch_data.py` (needs a Kaggle API token), then
`R/00_build.R` to produce the processed parquet, then `p1`/`p2`/`p3` in order.

## Data

[IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) — Vesta
card-not-present transactions, ~6 months, USD. `isFraud` is a chargeback-based label (a proxy,
not adjudicated truth), so the fraud cost is an upper bound; see the report's *Assumptions &
limitations* for the full caveats.

## Stack

R (tidyverse, **xgboost**, **glmnet**, **yardstick**, **arrow**), **Quarto**, **Shiny**,
`renv` for reproducibility; Python for data fetch.
