# app.R — Interactive sensitivity: drag the fraud/friction cost ratio k and watch
# the optimal block threshold and the savings move. Deploys to shinyapps.io.
# Reads the committed light artifact (scored.parquet); never calls Kaggle.
#
# Layout follows the house style of dev/equity_analysis: bslib page_sidebar, the
# same dark bs_theme (Inter, #4FC3F7 primary), card-based main panel, value boxes
# for the headline numbers, and a table for the savings breakdown.

library(shiny)
library(bslib)
library(ggplot2)
library(arrow)

# Resolve project files from either the project root (".") or one level up ("..").
# shiny::runApp("app") sets the working directory to app/, so root-relative paths
# need the "..". On shinyapps.io the bundle root is the project layout, so "." wins.
# Same lib.R / scored.parquet are used across every surface (single source of truth).
proj_path <- function(rel) {
  for (base in c(".", "..")) {
    p <- file.path(base, rel)
    if (file.exists(p)) return(p)
  }
  stop("cannot locate ", rel, " from ", normalizePath(getwd()))
}

source(proj_path(file.path("R", "lib.R")))
scored <- read_parquet(proj_path(file.path("data", "processed", "scored.parquet")))

# The k-free decomposition is computed ONCE at startup; the slider only recombines
# it as fn + fp/k (see lib.R). So dragging k never re-scans the 118k rows.
# Two decompositions: the production model (xgboost, "pd") and the interpretable
# Lasso baseline ("pd_logit"), so the curve overlays the two models' dollar cost.
components       <- cost_components(scored, pred = "pd")
components_logit <- cost_components(scored, pred = "pd_logit")

usd <- function(x) paste0("$", format(round(x), big.mark = ",", scientific = FALSE))

# --- Theme: mirror dev/equity_analysis ---------------------------------------
tema <- bs_theme(
  version      = 5,
  bg           = "#0f1117",
  fg           = "#e8eaf6",
  primary      = "#4FC3F7",
  secondary    = "#546e7a",
  success      = "#66BB6A",
  danger       = "#EF5350",
  base_font    = font_google("Inter"),
  heading_font = font_google("Inter"),
  font_scale   = 0.9
) |> bs_add_rules("
  .navbar { border-bottom: 1px solid #1e2130; }
  .sidebar { background: #13151f !important; border-right: 1px solid #1e2130; }
  .card { background: #13151f; border: 1px solid #1e2130; border-radius: 10px; }
  .card-header { background: transparent; border-bottom: 1px solid #1e2130; font-weight: 600; }
  .savings-table { width: 100%; border-collapse: collapse; }
  .savings-table th, .savings-table td { padding: 8px 12px; border-bottom: 1px solid #1e2130; text-align: right; }
  .savings-table th { color: #9aa0b5; font-weight: 600; }
  .savings-table td:first-child, .savings-table th:first-child { text-align: left; }
  .savings-table tr.optimum td { color: #4FC3F7; font-weight: 600; }
")

# ggplot dark theme matching the cards (thematic isn't installed, so set it here).
theme_dash <- function() {
  theme_minimal(base_size = 13) +
    theme(
      plot.background  = element_rect(fill = "#13151f", color = "#13151f"),
      panel.background = element_rect(fill = "#13151f", color = NA),
      panel.border     = element_blank(),
      panel.grid.major = element_line(color = "#1e2130"),
      panel.grid.minor = element_blank(),
      # ggplot2 4.0 draws axis lines by default -> the stray white rule under the
      # x-axis label. Blank every axis line/tick so only the grid shows.
      axis.line        = element_blank(),
      axis.ticks       = element_blank(),
      text             = element_text(color = "#e8eaf6"),
      axis.text        = element_text(color = "#9aa0b5"),
      legend.position   = "top",
      legend.title      = element_blank(),
      legend.key        = element_blank(),
      legend.background = element_rect(fill = "#13151f", color = NA)
    )
}

# =============================================================================
# UI
# =============================================================================
ui <- page_sidebar(
  title = "Fraud threshold — sensitivity to the cost ratio",
  theme = tema,
  sidebar = sidebar(
    width = 320,
    sliderInput("k", "Fraud cost / friction cost (k)",
                min = 5, max = 50, value = K_BASE, step = 1),
    helpText("k = how many dollars of fraud one blocked good customer is worth.",
             "Higher k -> fraud hurts more -> block more aggressively.")
  ),
  layout_columns(
    fill = FALSE,
    col_widths = c(4, 4, 4),
    value_box(title = "Optimal threshold", value = textOutput("vb_threshold"),
              theme = "primary"),
    value_box(title = "Total expected cost", value = textOutput("vb_cost")),
    value_box(title = "Saved vs 0.5 cutoff", value = textOutput("vb_savings"),
              theme = "success")
  ),
  card(
    card_header("Cost vs block threshold"),
    plotOutput("cost_curve", height = "380px")
  ),
  card(
    card_header("Savings breakdown — the optimum against its baselines"),
    uiOutput("savings_table")
  )
)

# =============================================================================
# Server
# =============================================================================
server <- function(input, output) {
  policy       <- reactive(optimal_policy(scored, input$k, components = components, pred = "pd"))
  policy_logit <- reactive(optimal_policy(scored, input$k, components = components_logit, pred = "pd_logit"))

  # ---- Value boxes ----------------------------------------------------------
  output$vb_threshold <- renderText(sprintf("%.3f", policy()$threshold))
  output$vb_cost      <- renderText(usd(policy()$cost))
  output$vb_savings   <- renderText(usd(policy()$savings_vs_block))

  # ---- Cost curve (ggplot, both models) ------------------------------------
  output$cost_curve <- renderPlot({
    k   <- input$k
    pol <- policy()
    cx  <- components;       cx$cost <- cx$fn_dollars + cx$fp_dollars / k
    cl  <- components_logit; cl$cost <- cl$fn_dollars + cl$fp_dollars / k
    df  <- rbind(
      data.frame(threshold = cx$threshold, cost = cx$cost, series = "xgboost"),
      data.frame(threshold = cl$threshold, cost = cl$cost, series = "Lasso baseline")
    )
    # Three legend entries share ONE color+linetype scale so the 0.5-cutoff line
    # appears in the legend alongside the two curves. Approve-all is a labeled
    # point (block nobody = threshold 1.0), not a line, so it stays off the legend.
    lv <- c("xgboost", "Lasso baseline", "0.5 cutoff")
    df$series <- factor(df$series, levels = lv)
    half_cost <- cx$cost[which.min(abs(cx$threshold - 0.5))]

    ggplot(df, aes(threshold, cost, color = series, linetype = series)) +
      geom_vline(aes(xintercept = 0.5, color = "0.5 cutoff", linetype = "0.5 cutoff"),
                 linewidth = 1.1) +
      geom_line(linewidth = 1) +
      # the off-the-shelf 0.5 cutoff and the economic optimum (on the xgboost curve)
      geom_point(aes(0.5, half_cost), color = "#FFD54F", size = 3,
                 inherit.aes = FALSE, show.legend = FALSE) +
      geom_point(aes(pol$threshold, pol$cost), color = "#EF5350", size = 3.5,
                 inherit.aes = FALSE, show.legend = FALSE) +
      # approve-all: block nobody -> cost = all fraud dollars, at threshold 1.0
      # where both curves converge. A white point + label, mirroring the optimum.
      geom_point(aes(1.0, pol$cost_approve_all), color = "#ECEFF1", size = 3.5,
                 inherit.aes = FALSE, show.legend = FALSE) +
      annotate("text", x = 1.0, y = pol$cost_approve_all, color = "#ECEFF1",
               label = "approve all", hjust = 1, vjust = -1, fontface = 2) +
      # label the optimum below its point; the bottom y-expansion (below) makes the
      # room so it neither clips nor collides with the approve-all point above it.
      annotate("text", x = pol$threshold, y = pol$cost, color = "#EF5350",
               label = sprintf("optimal t = %.3f", pol$threshold),
               hjust = 0.5, vjust = 1.9, fontface = 2) +
      scale_color_manual(name = NULL, limits = lv,
        values = c("xgboost" = "#4FC3F7", "Lasso baseline" = "#A5D6A7",
                   "0.5 cutoff" = "#FFD54F")) +
      scale_linetype_manual(name = NULL, limits = lv,
        values = c("xgboost" = 1, "Lasso baseline" = 1, "0.5 cutoff" = 2)) +
      scale_y_continuous(labels = function(v) paste0("$", round(v / 1e3), "k"),
                         expand = expansion(mult = c(0.22, 0.08))) +
      labs(x = "Block threshold (PD cutoff)", y = "Total expected cost",
           title = sprintf("k = %d", k)) +
      theme_dash()
  })

  # ---- Savings table --------------------------------------------------------
  output$savings_table <- renderUI({
    pol <- policy()
    pl  <- policy_logit()
    opt <- pol$cost
    row <- function(name, thr, cost, cls = "") {
      delta <- if (is.na(cost) || abs(cost - opt) < 1e-6) HTML("&mdash;")
               else paste0("+", usd(cost - opt))
      thr_cell <- if (is.na(thr)) HTML("&mdash;") else sprintf("%.3f", thr)
      tags$tr(class = cls,
        tags$td(name),
        tags$td(thr_cell),
        tags$td(usd(cost)),
        tags$td(delta))
    }
    tags$table(class = "savings-table",
      tags$thead(tags$tr(
        tags$th("Policy"), tags$th("Block threshold"),
        tags$th("Total cost"), tags$th("Extra cost vs optimum"))),
      tags$tbody(
        row("Economic optimum (xgboost)", pol$threshold, pol$cost, cls = "optimum"),
        row("0.5 cutoff (off-the-shelf)", 0.5, pol$cost_block_half),
        row("Approve everything", NA, pol$cost_approve_all),
        row("Lasso optimum (interpretable baseline)", pl$threshold, pl$cost)
      )
    )
  })
}

shinyApp(ui, server)
