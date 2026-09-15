# ==============================================================================
# MSc Business Analytics Dissertation - Queen's University Belfast
# Predicting Cybersecurity Threat Severity Using Machine Learning:
# A Critical Evaluation of Synthetic Dataset Validity in Business Analytics
#
# Author: Sanjai Balaji (40478904)
#
# PURPOSE
#   Reproduces every table (T01-T12) and figure (F01-F09) used in the
#   Research Report and Technical Report. Run top to bottom in R/RStudio.
#   All outputs are written to ./outputs/tables and ./outputs/figures.
#
# STRUCTURE (matches CRISP-DM phases referenced in the Methodology chapter)
#   0  Setup
#   1  Data Understanding      - load, inventory, JSON extraction
#   2  Data Quality            - missing, duplicates, constants, invalid values
#   3  Descriptive Stats       - T03, F01
#   4  Distribution Tests      - KS + chi-square uniformity, T04, F02
#   5  Correlation             - T05, F03
#   6  Internal Consistency    - T06, F04
#   7  Categorical Assoc.      - T07, F05
#   8  Predictive Modelling    - 5-fold CV, 4 targets x 4 models, T08
#   9  Feature Importance      - T09, F07
#   10 Permutation Test        - T10, F08
#   11 Learning Curve          - T11, F09
#   12 Classification Reframing- T12
#   13 Session Info            - reproducibility
#
# NOTE ON MEMORY
#   Random Forest / XGBoost are fitted on a fixed random sample of 10,000 rows
#   (seed 40478904) with capped tree depth to keep memory under ~2 GB. This is
#   documented as a design decision in the Technical Report. Statistical tests
#   (sections 3-7) use all 50,000 rows.
# ==============================================================================

# ---- 0. SETUP ----------------------------------------------------------------
pkgs <- c("stringr", "dplyr", "tidyr", "ggplot2", "randomForest", "rpart",
          "xgboost", "corrplot", "scales")
to_install <- pkgs[!pkgs %in% installed.packages()[, "Package"]]
if (length(to_install)) install.packages(to_install)
invisible(lapply(pkgs, library, character.only = TRUE))

set.seed(40478904)
dir.create("outputs", showWarnings = FALSE)
dir.create("outputs/tables", showWarnings = FALSE)
dir.create("outputs/figures", showWarnings = FALSE)

# >>> EDIT THIS PATH TO YOUR FILE <<<
DATA_PATH <- "path/to/cybersecurity_risk_dataset_50000.csv"

# Consistent figure style (Times New Roman to match the dissertation body text)
theme_diss <- theme_minimal(base_size = 11, base_family = "serif") +
  theme(plot.title = element_text(face = "bold", size = 12),
        panel.grid.minor = element_blank(),
        legend.position = "bottom")
theme_set(theme_diss)

save_fig <- function(p, name, w = 6.5, h = 4.2) {
  ggsave(file.path("outputs/figures", paste0(name, ".png")), p,
         width = w, height = h, dpi = 300, bg = "white")
}
save_tab <- function(x, name) {
  write.csv(x, file.path("outputs/tables", paste0(name, ".csv")), row.names = FALSE)
}
r2   <- function(actual, pred) 1 - sum((actual - pred)^2) / sum((actual - mean(actual))^2)
rmse <- function(actual, pred) sqrt(mean((actual - pred)^2))

# ---- 1. DATA UNDERSTANDING -----------------------------------------------------
df_raw <- read.csv(DATA_PATH, stringsAsFactors = FALSE)
cat("Loaded:", nrow(df_raw), "rows x", ncol(df_raw), "columns\n")

# T01 - Column inventory: type, cardinality, missingness
inventory <- data.frame(
  column     = names(df_raw),
  type       = sapply(df_raw, function(x) class(x)[1]),
  n_unique   = sapply(df_raw, function(x) length(unique(x))),
  n_missing  = sapply(df_raw, function(x) sum(is.na(x) | x == "")),
  row.names  = NULL, stringsAsFactors = FALSE
)
inventory$pct_unique <- round(100 * inventory$n_unique / nrow(df_raw), 1)
inventory$is_nested_json <- grepl("^\\{|^\\[", sapply(df_raw, function(x) as.character(x[1])))
save_tab(inventory, "T01_column_inventory")
print(inventory)

# Extract numeric fields embedded in JSON-like text columns
# (Python's jsonlite parsing failed here: the fields use single-quoted Python
#  dict syntax, not valid JSON, so a lookbehind-regex extractor was used instead.)
df <- df_raw
ext_num <- function(col, key) as.numeric(str_extract(col, paste0("(?<='", key, "': )[0-9.]+")))

df$hw_count          <- ext_num(df$asset_count, "hardware")
df$sw_count          <- ext_num(df$asset_count, "software")
df$critical_assets   <- ext_num(df$asset_count, "critical_assets")
df$revenue_per_hour  <- ext_num(df$business_context, "revenue_per_hour")
df$revenue_annual    <- ext_num(df$business_context, "revenue_annual")
df$data_value_per_gb <- ext_num(df$business_context, "data_value_per_gb")
df$industry_sector   <- str_extract(df$business_context, "(?<='industry_sector': ')[A-Za-z ]+")
df$controls_score    <- ext_num(df$existing_controls, "existing_controls_score")
df$preventive        <- ext_num(df$existing_controls, "preventive")
df$detective          <- ext_num(df$existing_controls, "detective")
df$prob_occurrence   <- ext_num(df$prediction_results, "probability_of_occurrence")

cat("After JSON extraction:", ncol(df), "columns\n")

# Variable groups used throughout
NUM_VARS <- c("threat_intelligence_score", "hardware_risk_avg", "software_risk_avg",
              "network_segment_risk", "data_classification_risk",
              "avg_days_since_last_similar", "hw_count", "sw_count", "critical_assets",
              "revenue_per_hour", "revenue_annual", "data_value_per_gb",
              "controls_score", "preventive", "detective", "prob_occurrence")
CAT_VARS <- c("scenario_type", "threat_type", "attack_vector", "attack_complexity",
              "initial_access_method", "industry_sector")
for (v in CAT_VARS) df[[v]] <- as.factor(df[[v]])
TARGETS <- c("threat_intelligence_score", "prob_occurrence", "controls_score", "critical_assets")

# ---- 2. DATA QUALITY -------------------------------------------------------------
const_cols   <- names(df)[sapply(df, function(x) is.numeric(x) && sd(x, na.rm = TRUE) == 0)]
invalid_prob <- sum(df$prob_occurrence > 1, na.rm = TRUE)

# T02 - Data quality summary
dq <- data.frame(
  check = c("Total records", "Raw columns", "Columns after JSON extraction",
            "Duplicate rows", "Numeric columns with missing values",
            "Constant (zero-variance) numeric columns",
            "High-cardinality text columns (>90% unique)",
            "prob_occurrence values > 1.0 (invalid probability)",
            "prob_occurrence maximum observed"),
  result = c(nrow(df), ncol(df_raw), ncol(df),
             sum(duplicated(df_raw)),
             sum(sapply(df[NUM_VARS], function(x) any(is.na(x)))),
             ifelse(length(const_cols) == 0, "none", paste(const_cols, collapse = "; ")),
             paste(inventory$column[inventory$pct_unique > 90], collapse = "; "),
             invalid_prob,
             round(max(df$prob_occurrence, na.rm = TRUE), 3)),
  stringsAsFactors = FALSE
)
save_tab(dq, "T02_data_quality")
print(dq)

# ---- 3. DESCRIPTIVE STATISTICS ----------------------------------------------------
desc <- df %>%
  select(all_of(NUM_VARS)) %>%
  pivot_longer(everything(), names_to = "variable") %>%
  group_by(variable) %>%
  summarise(n = sum(!is.na(value)),
            mean = mean(value, na.rm = TRUE), sd = sd(value, na.rm = TRUE),
            min = min(value, na.rm = TRUE), q25 = quantile(value, .25, na.rm = TRUE),
            median = median(value, na.rm = TRUE), q75 = quantile(value, .75, na.rm = TRUE),
            max = max(value, na.rm = TRUE), .groups = "drop") %>%
  mutate(across(where(is.numeric), ~ round(.x, 4)))
save_tab(desc, "T03_descriptive_statistics")
print(desc, n = Inf)

# F01 - Faceted histograms of the six [0.5,1] bounded scores.
# Dashed line = expected count per bin under a perfectly uniform distribution.
score_vars <- c("threat_intelligence_score", "hardware_risk_avg", "software_risk_avg",
                "network_segment_risk", "data_classification_risk", "controls_score")
long_scores <- df %>% select(all_of(score_vars)) %>%
  pivot_longer(everything(), names_to = "variable", values_to = "value")
bins <- 25
exp_per_bin <- nrow(df) / bins
p1 <- ggplot(long_scores, aes(value)) +
  geom_histogram(bins = bins, fill = "steelblue", colour = "white", linewidth = .2) +
  geom_hline(yintercept = exp_per_bin, linetype = "dashed", colour = "firebrick") +
  facet_wrap(~ variable, ncol = 3) +
  labs(title = "Distribution of bounded score variables (n = 50,000)",
       subtitle = "Dashed line = expected count per bin under a uniform(0.5, 1.0) distribution",
       x = NULL, y = "Count")
save_fig(p1, "F01_score_distributions", h = 4.8)

# ---- 4. DISTRIBUTION (UNIFORMITY) TESTS -------------------------------------------
# Two independent tests per variable:
#   (a) One-sample Kolmogorov-Smirnov vs U(a,b)         (note: ties warning is benign)
#   (b) Chi-square goodness-of-fit over 20 equal-width bins vs uniform expected
unif_test <- function(x, a, b, nb = 20) {
  x <- x[!is.na(x) & x >= a & x <= b]
  ks <- suppressWarnings(ks.test(x, "punif", a, b))
  obs <- table(cut(x, breaks = seq(a, b, length.out = nb + 1), include.lowest = TRUE))
  cs <- chisq.test(obs, p = rep(1 / nb, nb))
  c(n = length(x), ks_D = unname(ks$statistic), ks_p = ks$p.value,
    chisq = unname(cs$statistic), chisq_p = cs$p.value)
}

unif_specs <- list(
  threat_intelligence_score   = c(0.5, 1), hardware_risk_avg = c(0.5, 1),
  software_risk_avg           = c(0.5, 1), network_segment_risk = c(0.5, 1),
  data_classification_risk    = c(0.5, 1), controls_score = c(0.5, 1),
  preventive                  = c(0.5, 1), detective = c(0.5, 1),
  prob_occurrence             = c(0, 1),        # values >1 excluded (see T02)
  avg_days_since_last_similar = c(30, 364),
  revenue_per_hour            = c(1000, 10000), data_value_per_gb = c(500, 2000)
)

# T04
unif_res <- do.call(rbind, lapply(names(unif_specs), function(v) {
  r <- unif_test(df[[v]], unif_specs[[v]][1], unif_specs[[v]][2])
  data.frame(variable = v, lower = unif_specs[[v]][1], upper = unif_specs[[v]][2],
             t(round(r, 4)))
}))
unif_res$uniform_not_rejected_05 <- unif_res$ks_p > 0.05 & unif_res$chisq_p > 0.05
save_tab(unif_res, "T04_uniformity_tests")
print(unif_res)

# F02 - Empirical CDF vs theoretical uniform CDF for the target variable
x <- sort(df$threat_intelligence_score)
p2 <- ggplot(data.frame(x = x, ecdf = seq_along(x) / length(x)), aes(x)) +
  geom_step(aes(y = ecdf, colour = "Empirical CDF"), linewidth = .6) +
  geom_line(aes(y = punif(x, 0.5, 1), colour = "Uniform(0.5, 1) CDF"),
            linetype = "dashed", linewidth = .6) +
  scale_colour_manual(values = c("steelblue", "firebrick"), name = NULL) +
  labs(title = "Empirical vs theoretical uniform CDF: threat_intelligence_score",
       x = "threat_intelligence_score", y = "Cumulative probability")
save_fig(p2, "F02_ecdf_target")

# ---- 5. CORRELATION ANALYSIS ------------------------------------------------------
cor_vars <- setdiff(NUM_VARS, const_cols)  # drop zero-variance columns
cm <- cor(df[cor_vars], use = "pairwise.complete.obs")

# T05 - full matrix + summary of off-diagonal magnitudes
save_tab(data.frame(variable = rownames(cm), round(cm, 4)), "T05_correlation_matrix")
off <- cm[upper.tri(cm)]
cor_summary <- data.frame(
  n_pairs = length(off), max_abs_r = round(max(abs(off)), 4),
  mean_abs_r = round(mean(abs(off)), 4),
  n_pairs_abs_r_gt_0.05 = sum(abs(off) > 0.05),
  target_max_abs_r = round(max(abs(cm["threat_intelligence_score",
                                       setdiff(cor_vars, "threat_intelligence_score")])), 4)
)
save_tab(cor_summary, "T05b_correlation_summary")
print(cor_summary)

# F03 - Heatmap (colour scale fixed at [-1,1] so near-zero values read as grey,
# rather than being auto-scaled into looking dramatic)
png("outputs/figures/F03_correlation_heatmap.png", width = 2000, height = 1800, res = 300)
corrplot(cm, method = "color", type = "upper", addCoef.col = "black",
         number.cex = .45, tl.cex = .6, tl.col = "black", col.lim = c(-1, 1),
         title = "Pearson correlation matrix (all numeric variables, n = 50,000)",
         mar = c(0, 0, 2, 0))
dev.off()

# ---- 6. INTERNAL CONSISTENCY ------------------------------------------------------
# The composite existing_controls_score is nested ABOVE preventive/detective in
# the JSON. If genuine, it must be a function of them. Test that claim directly.
ic_mod <- lm(controls_score ~ preventive + detective, data = df)
ic_sum <- summary(ic_mod)

# T06
ic_tab <- data.frame(
  term      = rownames(ic_sum$coefficients),
  estimate  = round(ic_sum$coefficients[, 1], 5),
  std_error = round(ic_sum$coefficients[, 2], 5),
  t_value   = round(ic_sum$coefficients[, 3], 3),
  p_value   = round(ic_sum$coefficients[, 4], 4), row.names = NULL
)
ic_tab <- rbind(ic_tab, data.frame(
  term = c("R-squared", "Adj R-squared", "F p-value"),
  estimate = round(c(ic_sum$r.squared, ic_sum$adj.r.squared,
                      pf(ic_sum$fstatistic[1], ic_sum$fstatistic[2],
                         ic_sum$fstatistic[3], lower.tail = FALSE)), 6),
  std_error = NA, t_value = NA, p_value = NA
))
save_tab(ic_tab, "T06_internal_consistency")
print(ic_tab)

# F04 - Composite vs components (sample 5,000 pts for legibility)
ic_long <- df %>% slice_sample(n = 5000) %>%
  select(controls_score, preventive, detective) %>%
  pivot_longer(-controls_score, names_to = "component", values_to = "component_score")
p4 <- ggplot(ic_long, aes(component_score, controls_score)) +
  geom_point(alpha = .15, size = .6, colour = "steelblue") +
  geom_smooth(method = "lm", colour = "firebrick", se = TRUE, linewidth = .8) +
  facet_wrap(~ component) +
  labs(title = "Composite controls score vs its stated components",
       subtitle = sprintf("OLS R-squared = %.5f | no coefficient significant at 5%%",
                           ic_sum$r.squared),
       x = "Component score", y = "existing_controls_score")
save_fig(p4, "F04_internal_consistency")

# ---- 7. CATEGORICAL ASSOCIATION ---------------------------------------------------
# One-way ANOVA + eta-squared: does ANY categorical variable explain variance
# in the target? (Complements the numeric-only correlation matrix.)

# T07
cat_res <- do.call(rbind, lapply(CAT_VARS, function(v) {
  a <- anova(lm(threat_intelligence_score ~ df[[v]], data = df))
  data.frame(variable = v, levels = nlevels(df[[v]]),
             F = round(a$`F value`[1], 3), p_value = round(a$`Pr(>F)`[1], 4),
             eta_squared = round(a$`Sum Sq`[1] / sum(a$`Sum Sq`), 6))
}))
save_tab(cat_res, "T07_categorical_association")
print(cat_res)

# F05 - Target by threat_type and attack_vector
p5a <- ggplot(df, aes(threat_type, threat_intelligence_score)) +
  geom_boxplot(fill = "steelblue", alpha = .6, outlier.size = .3) +
  coord_flip() + labs(title = "Threat score by threat type", x = NULL, y = NULL)
p5b <- ggplot(df, aes(attack_vector, threat_intelligence_score)) +
  geom_boxplot(fill = "darkorange", alpha = .6, outlier.size = .3) +
  coord_flip() + labs(title = "Threat score by attack vector", x = NULL, y = NULL)
save_fig(p5a, "F05a_target_by_threat_type", h = 3.6)
save_fig(p5b, "F05b_target_by_attack_vector", h = 3.2)

# ---- 8. PREDICTIVE MODELLING (5-fold CV, 4 targets x 4 models) --------------------
FEATURES <- c(CAT_VARS[CAT_VARS != "scenario_type"],
              setdiff(NUM_VARS, c(TARGETS, const_cols)))

# Target leakage verification - run once.
# (The first run of this pipeline accidentally left the target inside FEATURES,
#  which produced R2 = 1.000 - investigated rather than reported. See README.)
cat("\nTarget leakage check:\n")
for (target_check in TARGETS) {
  cat(target_check, "present in FEATURES:", target_check %in% FEATURES, "\n")
}

samp <- df[sample(nrow(df), 10000), ]
K <- 5

fit_predict <- function(model, train, test) {
  switch(model,
    "Linear Regression" = predict(lm(y ~ ., train), test),
    "Decision Tree"     = predict(rpart(y ~ ., train, method = "anova"), test),
    "Random Forest"     = predict(randomForest(y ~ ., train, ntree = 100,
                                                nodesize = 20, maxnodes = 200), test),
    "XGBoost" = {
      mm_tr <- model.matrix(y ~ . - 1, train)
      mm_te <- model.matrix(y ~ . - 1, test)
      bst <- xgboost(data = mm_tr, label = train$y, nrounds = 200,
                      max_depth = 4, eta = 0.05, subsample = .8,
                      colsample_bytree = .8, objective = "reg:squarederror", verbose = 0)
      predict(bst, mm_te)
    })
}

MODELS <- c("Linear Regression", "Decision Tree", "Random Forest", "XGBoost")
cv_results <- list()

for (tg in TARGETS) {
  preds <- setdiff(FEATURES, tg)
  d <- samp[, c(preds, tg)]
  d <- d[complete.cases(d), ]
  names(d)[ncol(d)] <- "y"
  folds <- sample(rep(1:K, length.out = nrow(d)))

  for (m in MODELS) {
    r2s <- rmses <- numeric(K)
    for (k in 1:K) {
      tr <- d[folds != k, ]
      te <- d[folds == k, ]
      p <- fit_predict(m, tr, te)
      r2s[k]   <- r2(te$y, p)
      rmses[k] <- rmse(te$y, p)
    }
    # Naive baseline: predict the training-fold mean. Gives negative R2 a
    # concrete reference point rather than reporting it in isolation.
    base_rmse <- mean(sapply(1:K, function(k) rmse(d$y[folds == k], mean(d$y[folds != k]))))
    cv_results[[length(cv_results) + 1]] <- data.frame(
      target = tg, model = m,
      cv_r2_mean = round(mean(r2s), 4), cv_r2_sd = round(sd(r2s), 4),
      cv_rmse_mean = round(mean(rmses), 5), baseline_rmse_mean = round(base_rmse, 5),
      rmse_vs_baseline_pct = round(100 * (mean(rmses) - base_rmse) / base_rmse, 2)
    )
    cat(sprintf("%-26s %-18s CV R2 = %7.4f (sd %.4f)\n", tg, m, mean(r2s), sd(r2s)))
  }
}
cv_tab <- do.call(rbind, cv_results)
save_tab(cv_tab, "T08_cv_model_results")

# ---- 9. FEATURE IMPORTANCE (primary target, full model) ---------------------------
d0 <- samp[, c(setdiff(FEATURES, "threat_intelligence_score"), "threat_intelligence_score")]
d0 <- d0[complete.cases(d0), ]; names(d0)[ncol(d0)] <- "y"
rf0 <- randomForest(y ~ ., d0, ntree = 200, nodesize = 20, maxnodes = 200, importance = TRUE)
imp <- as.data.frame(importance(rf0)); imp$feature <- rownames(imp)
imp <- imp %>% arrange(desc(`%IncMSE`)) %>% mutate(across(where(is.numeric), ~ round(.x, 4)))
save_tab(imp, "T09_rf_feature_importance")

p7 <- ggplot(imp, aes(reorder(feature, `%IncMSE`), `%IncMSE`)) +
  geom_col(fill = "steelblue") + coord_flip() +
  labs(title = "Random Forest permutation importance (%IncMSE), target = threat_intelligence_score",
       subtitle = "Values near/below zero indicate no useful predictive contribution",
       x = NULL, y = "% increase in MSE when permuted")
save_fig(p7, "F07_rf_feature_importance", h = 4.8)

# ---- 10. PERMUTATION (LABEL-SHUFFLE) TEST ------------------------------------------
# If the model performs the same when the target is randomly shuffled, the
# "real" model has learned nothing beyond noise.
perm_test <- function(d, n_perm = 10) {
  idx <- sample(nrow(d), round(.8 * nrow(d))); tr <- d[idx, ]; te <- d[-idx, ]
  real <- r2(te$y, predict(randomForest(y ~ ., tr, ntree = 100, nodesize = 20,
                                         maxnodes = 200), te))
  perms <- replicate(n_perm, {
    trs <- tr; trs$y <- sample(trs$y)
    r2(te$y, predict(randomForest(y ~ ., trs, ntree = 100, nodesize = 20, maxnodes = 200), te))
  })
  c(real_r2 = real, perm_mean_r2 = mean(perms), perm_sd_r2 = sd(perms),
    p_value = mean(perms >= real))
}
pt <- round(perm_test(d0), 4)
perm_tab <- data.frame(target = "threat_intelligence_score", t(pt))
save_tab(perm_tab, "T10_permutation_test")
print(perm_tab)

# F08
p8 <- ggplot(data.frame(condition = c("Real target", "Shuffled target (mean of 10)"),
                         r2 = c(pt["real_r2"], pt["perm_mean_r2"])),
              aes(condition, r2, fill = condition)) +
  geom_col(width = .5) + geom_hline(yintercept = 0, linetype = "dashed") +
  scale_fill_manual(values = c("steelblue", "grey60")) + theme(legend.position = "none") +
  labs(title = "Permutation test: Random Forest R-squared, real vs shuffled target",
       subtitle = sprintf("Permutation p-value = %.2f", pt["p_value"]),
       x = NULL, y = "Test R-squared")
save_fig(p8, "F08_permutation_test", h = 3.6)

# ---- 11. LEARNING CURVE ------------------------------------------------------------
# Does more data help? For a genuine signal, R2 should rise with n. For noise,
# it stays flat around zero regardless of sample size.
lc <- do.call(rbind, lapply(c(500, 1000, 2500, 5000, 10000), function(n) {
  dd <- d0[sample(nrow(d0), min(n, nrow(d0))), ]
  idx <- sample(nrow(dd), round(.8 * nrow(dd)))
  m <- randomForest(y ~ ., dd[idx, ], ntree = 100, nodesize = 20, maxnodes = 200)
  data.frame(n_train = length(idx),
             test_r2 = round(r2(dd$y[-idx], predict(m, dd[-idx, ])), 4))
}))
save_tab(lc, "T11_learning_curve")

p9 <- ggplot(lc, aes(n_train, test_r2)) + geom_line(colour = "steelblue") +
  geom_point(size = 2, colour = "steelblue") + geom_hline(yintercept = 0, linetype = "dashed") +
  scale_x_continuous(labels = comma) +
  labs(title = "Learning curve: Random Forest test R-squared vs training size",
       x = "Training observations", y = "Test R-squared")
save_fig(p9, "F09_learning_curve", h = 3.6)

# ---- 12. CLASSIFICATION RE-FRAMING TEST --------------------------------------------
# Would binning severity into Low/Medium/High (rather than predicting it as a
# continuous score) recover any signal? Tested at the supervisor's suggestion.
d0$y_class <- cut(d0$y, breaks = quantile(d0$y, c(0, 1/3, 2/3, 1)),
                   labels = c("Low", "Medium", "High"), include.lowest = TRUE)
idx <- sample(nrow(d0), round(.8 * nrow(d0)))
rfc <- randomForest(y_class ~ . - y, d0[idx, ], ntree = 100, nodesize = 20, maxnodes = 200)
pred_c <- predict(rfc, d0[-idx, ])
acc <- mean(pred_c == d0$y_class[-idx])
majority <- max(prop.table(table(d0$y_class[-idx])))
cls_tab <- data.frame(metric = c("RF 3-class accuracy", "Majority-class baseline",
                                  "Chance (3 balanced classes)"),
                       value = round(c(acc, majority, 1/3), 4))
save_tab(cls_tab, "T12_classification_reframing")
print(cls_tab)
print(table(predicted = pred_c, actual = d0$y_class[-idx]))

# ---- 13. SESSION INFO (reproducibility, appendix of Technical Report) --------------
writeLines(capture.output(sessionInfo()), "outputs/session_info.txt")
cat("\nDONE. Tables in outputs/tables, figures in outputs/figures.\n")
