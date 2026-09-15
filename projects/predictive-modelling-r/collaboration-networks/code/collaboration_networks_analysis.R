# ==============================================================================
# ITAO7103 Assignment 1 — Predicting Star Performance in Collaboration Networks
# MSc Business Analytics, Queen's University Belfast
# Author: Sanjai Balaji (40478904)
#
# Tests three competing theories of high performance in collaboration networks:
#   - elite learning (Liu et al., 2018)
#   - structural holes / brokerage (Burt, 2004)
#   - technological breadth (Fleming, 2001)
# via EDA, linear/polynomial regression, LASSO, and classification (logistic
# regression vs LDA), evaluated with VIF, standardised coefficients, and ROC/AUC.
# ==============================================================================

library(readxl)
library(dplyr)
library(ggplot2)
library(corrplot)
library(car)          # vif()
library(glmnet)
library(lm.beta)
library(caret)
library(MASS)          # lda()
library(pROC)
library(patchwork)      # p1 + p2 + p3 layout
library(broom)          # glance()

# ---- 1. Load and prepare data ---------------------------------------------------
DATA_PATH <- "path/to/advanced_analytics_dataset.xlsx"
df <- read_excel(DATA_PATH)

df <- df %>%
  mutate(
    Is_HighPerformer = as.factor(Is_HighPerformer),
    Log_Impact = log(Total_Impact + 1)
  )

colSums(is.na(df))
df <- na.omit(df)          # complete-case analysis (see README for rationale)
colSums(is.na(df))
cat("Rows:", nrow(df), "Columns:", ncol(df))

# ---- 2. Exploratory Data Analysis ------------------------------------------------

# 2.1 Summary statistics
vars <- c("Total_Impact", "Total_Projects", "Career_Length",
          "Num_Elite_Collaborators", "Num_Regular_Collaborators",
          "Network_Constraint", "Effective_Size")
summary_table <- df %>% select(all_of(vars)) %>% summary()
print(summary_table)

# 2.2 Correlation matrix
num_data <- df %>% select(where(is.numeric)) %>% na.omit()
corr_matrix <- cor(num_data, use = "complete.obs")
corrplot(
  corr_matrix, method = "color", type = "upper", order = "hclust",
  addCoef.col = "black", tl.col = "black", tl.cex = 1.1, number.cex = 0.7,
  col = colorRampPalette(c("darkblue", "white", "darkred"))(200), diag = FALSE
)

# 2.3 Scatter plots
p1 <- ggplot(df, aes(Num_Elite_Collaborators, Log_Impact)) +
  geom_point(alpha = 0.2) + geom_smooth(method = "lm", color = "red") + theme_minimal()
p2 <- ggplot(df, aes(Cohesion_Elite, Log_Impact)) +
  geom_point(alpha = 0.2) + geom_smooth(method = "lm", color = "red") + theme_minimal()
p3 <- ggplot(df, aes(Effective_Size, Log_Impact)) +
  geom_point(alpha = 0.2) + geom_smooth(method = "lm", color = "red") + theme_minimal()
p1 + p2 + p3

# 2.4 Boxplots / violin plots by performance status
ggplot(df, aes(x = Is_HighPerformer, y = Effective_Size, fill = Is_HighPerformer)) +
  geom_boxplot(alpha = 0.7, outlier.alpha = 0.2) +
  scale_fill_manual(values = c("#2C7BB6", "#D7191C")) +
  theme_classic() +
  labs(title = "Network Brokerage (Effective Size) by Performance Status",
       x = "High Performer", y = "Effective Size")

ggplot(df, aes(x = Is_HighPerformer, y = Num_Elite_Collaborators, fill = Is_HighPerformer)) +
  geom_violin(trim = FALSE, alpha = 0.6) +
  geom_boxplot(width = 0.15, color = "black", outlier.alpha = 0.3) +
  theme_minimal() +
  labs(title = "Distribution of Elite Collaborations by Performance Status",
       x = "High Performer Status", y = "Number of Elite Collaborators")

# ---- 3. Linear Regression ---------------------------------------------------------

# 3.1 Simple regression
m1 <- lm(Log_Impact ~ Num_Elite_Collaborators, data = df)
summary(m1)
par(mfrow = c(2, 2)); plot(m1)

# 3.2 Multiple regression
m2 <- lm(Log_Impact ~ Total_Projects + Career_Length +
            Num_Elite_Collaborators + Cohesion_Elite +
            Effective_Size, data = df)
summary(m2)
vif(m2)
par(mfrow = c(2, 2)); plot(m2)

# 3.3 Polynomial (career-length accumulation effect)
m3 <- lm(Log_Impact ~ Career_Length + I(Career_Length^2), data = df)
summary(m3)

# 3.4 Standardised coefficients (relative importance)
lm.beta(m2)

# ---- 4. LASSO variable selection ---------------------------------------------------
x <- model.matrix(Log_Impact ~ . - Total_Impact - Is_HighPerformer, df)[, -1]
y <- df$Log_Impact

set.seed(40478904)
lasso_cv <- cv.glmnet(x, y, alpha = 1)
plot(lasso_cv)

best_lambda <- lasso_cv$lambda.min
lasso_model <- lasso_cv
coef(lasso_model, s = "lambda.min")
coef(lasso_model)   # lambda.1se (more parsimonious)

forward_model <- step(
  lm(Log_Impact ~ . - Total_Impact - Is_HighPerformer - AID, data = df),
  direction = "forward"
)
summary(forward_model)

# ---- 5. Classification: Logistic Regression vs LDA ---------------------------------
set.seed(40478904)
train_index <- createDataPartition(df$Is_HighPerformer, p = 0.7, list = FALSE)
train <- df[train_index, ]
test  <- df[-train_index, ]

log_model <- glm(Is_HighPerformer ~ Num_Elite_Collaborators +
                    Cohesion_Elite + Effective_Size + Career_Length,
                  data = train, family = binomial)
pred_prob  <- predict(log_model, test, type = "response")
pred_class <- ifelse(pred_prob > 0.5, 1, 0)
table(pred_prob > 0.3)
confusionMatrix(as.factor(pred_class), as.factor(test$Is_HighPerformer))

lda_model <- lda(Is_HighPerformer ~ Num_Elite_Collaborators +
                    Cohesion_Elite + Effective_Size + Career_Length,
                  data = train)
lda_pred <- predict(lda_model, test)
confusionMatrix(as.factor(lda_pred$class), as.factor(test$Is_HighPerformer))

# ---- 6. ROC Curve Comparison ---------------------------------------------------------
roc_log <- roc(as.numeric(test$Is_HighPerformer), pred_prob)
lda_prob <- predict(lda_model, test)$posterior[, 2]
roc_lda <- roc(as.numeric(test$Is_HighPerformer), lda_prob)

plot(roc_log, col = "blue", main = "ROC Curve Comparison")
plot(roc_lda, add = TRUE, col = "red")
legend("bottomright", legend = c("Logistic Regression", "LDA"),
       col = c("blue", "red"), lwd = 2)

auc(roc_log)
auc(roc_lda)
coords(roc_log, "best", ret = "threshold")

# ---- 7. Interaction effect: does cohesion amplify elite learning? -------------------
interaction_model <- glm(Is_HighPerformer ~ Num_Elite_Collaborators * Cohesion_Elite +
                            Effective_Size + Career_Length,
                          data = train, family = binomial)
summary(interaction_model)  # interaction term not significant (p = 0.283)

# ---- Appendix: additional diagnostics -------------------------------------------------
# A1 - distribution of the (log-transformed) dependent variable
ggplot(df, aes(x = Log_Impact)) +
  geom_histogram(bins = 40, fill = "steelblue", alpha = 0.7) +
  theme_minimal() +
  labs(title = "Distribution of Log(Impact)", x = "Log Impact", y = "Frequency")

# A2 - ellipse correlation plot, core structural variables only
num_vars <- df %>%
  select(Total_Projects, Career_Length, Num_Elite_Collaborators,
         Num_Regular_Collaborators, Network_Constraint, Effective_Size, Log_Impact)
corrplot(cor(num_vars), method = "ellipse", type = "upper", tl.cex = 0.8, addCoef.col = "black")

# A3 - model comparison table (R2 / adjusted R2 / AIC across the three linear models)
model_stats <- bind_rows(
  glance(m1) %>% mutate(model = "Simple LM"),
  glance(m2) %>% mutate(model = "Multiple LM"),
  glance(m3) %>% mutate(model = "Polynomial LM")
)
model_stats %>% select(model, r.squared, adj.r.squared, AIC)
