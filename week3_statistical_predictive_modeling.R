# Week 3: Statistical Analysis and Predictive Modeling using R
# Dataset: built-in iris dataset (publicly documented dataset)
# Task: Binary classification — Setosa vs Non-setosa

library(dplyr)
library(caret)
library(pROC)

set.seed(42)
data("iris")

# Create binary target
iris_model <- iris %>%
  mutate(Setosa = ifelse(Species == "setosa", 1, 0))

# 1. Dataset overview
str(iris_model)
summary(iris_model)

# 2. Hypothesis testing
# H0: mean Petal.Length is equal for Setosa and non-Setosa
# H1: mean Petal.Length differs
print(t.test(Petal.Length ~ Species, data =
  iris %>% mutate(Group = ifelse(Species == "setosa", "Setosa", "Non-setosa")) %>%
  filter(Group %in% c("Setosa", "Non-setosa"))))

# Correlation between petal measurements
print(cor.test(iris$Petal.Length, iris$Petal.Width, method = "pearson"))

# Normality check
print(shapiro.test(iris$Petal.Length))

# 3. Train/test split
set.seed(42)
idx <- createDataPartition(iris_model$Setosa, p = 0.80, list = FALSE)
train <- iris_model[idx, ]
test <- iris_model[-idx, ]

# 4. Logistic regression
model <- glm(Setosa ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width,
             data = train, family = binomial())
summary(model)

# 5. Test predictions and classification metrics
prob <- predict(model, newdata = test, type = "response")
pred <- ifelse(prob >= 0.50, 1, 0)

cm <- confusionMatrix(
  factor(pred, levels = c(0,1)),
  factor(test$Setosa, levels = c(0,1)),
  positive = "1"
)
print(cm)

# 6. ROC / AUC
roc_obj <- roc(test$Setosa, prob)
print(auc(roc_obj))
plot(roc_obj, main = "ROC Curve — Logistic Regression")

# 7. Model diagnostics: residuals vs probability
residuals_df <- data.frame(
  predicted = prob,
  residual = test$Setosa - prob
)
plot(residuals_df$predicted, residuals_df$residual,
     xlab = "Predicted probability",
     ylab = "Observed - predicted",
     main = "Classification Residuals vs Predicted Probability")
abline(h = 0)

# 8. Five-fold cross-validation
set.seed(42)
ctrl <- trainControl(method = "cv", number = 5)
cv_model <- train(
  factor(Setosa) ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width,
  data = iris_model,
  method = "glm",
  family = binomial(),
  trControl = ctrl,
  metric = "Accuracy"
)
print(cv_model)

# 9. Potential improvements
# - Compare with decision tree / random forest.
# - Tune the probability threshold for different business costs.
# - Add interaction terms and evaluate calibration.
