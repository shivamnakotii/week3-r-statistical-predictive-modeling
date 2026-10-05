# WEEK 4 - Comprehensive Data Analysis Reporting
# Shivam Nakoti
# This script consolidates the methods used across Weeks 1-3.

# Packages
library(tidyverse)
library(ggplot2)
library(dplyr)
library(scales)
library(caret)
library(pROC)

# ============================================================
# WEEK 1: Titanic - Data Preparation and Exploratory Analysis
# ============================================================
# Public Titanic CSV
titanic <- read.csv("titanic.csv", stringsAsFactors = FALSE)

# Missing-value profile
colSums(is.na(titanic))

# Remove exact duplicate rows
titanic <- distinct(titanic)

# Basic cleaning / imputation example
titanic$Age[is.na(titanic$Age)] <- median(titanic$Age, na.rm = TRUE)
titanic$Embarked[is.na(titanic$Embarked) | titanic$Embarked == ""] <-
  names(sort(table(titanic$Embarked), decreasing = TRUE))[1]

# Descriptive statistics
summary(titanic)

# Correlation among selected numeric variables
num_vars <- titanic %>% select(where(is.numeric))
cor(num_vars, use = "pairwise.complete.obs")

# IQR outlier rule for Fare
Q1 <- quantile(titanic$Fare, .25, na.rm = TRUE)
Q3 <- quantile(titanic$Fare, .75, na.rm = TRUE)
IQR_value <- IQR(titanic$Fare, na.rm = TRUE)
fare_outliers <- titanic %>%
  filter(Fare < Q1 - 1.5*IQR_value | Fare > Q3 + 1.5*IQR_value)

# ============================================================
# WEEK 2: Diamonds - Visualization and Insight Communication
# ============================================================
data("diamonds", package = "ggplot2")

# Distribution
ggplot(diamonds, aes(price)) +
  geom_histogram(bins = 40, fill = "#2F6BFF", color = "white") +
  scale_x_continuous(labels = comma) +
  theme_minimal()

# Relationship
ggplot(diamonds, aes(carat, price, color = cut)) +
  geom_point(alpha = .25, size = 1) +
  scale_y_continuous(labels = comma) +
  theme_minimal()

# Category comparison
cut_summary <- diamonds %>%
  group_by(cut) %>%
  summarise(avg_price = mean(price), .groups = "drop")

ggplot(cut_summary, aes(reorder(cut, avg_price), avg_price)) +
  geom_col(fill = "#FF7A00") +
  coord_flip() +
  scale_y_continuous(labels = comma) +
  theme_minimal()

# Spread
ggplot(diamonds, aes(clarity, price)) +
  geom_boxplot(alpha = .75) +
  scale_y_continuous(labels = comma) +
  theme_minimal()

# Trend
trend <- diamonds %>%
  mutate(carat_range = cut(carat, breaks = seq(0, 5, .25),
                           include.lowest = TRUE)) %>%
  group_by(carat_range) %>%
  summarise(avg_price = mean(price), .groups = "drop") %>%
  filter(!is.na(avg_price))

ggplot(trend, aes(carat_range, avg_price, group = 1)) +
  geom_line(linewidth = 1) +
  geom_point() +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# ============================================================
# WEEK 3: Iris - Statistical Analysis and Predictive Modeling
# ============================================================
data(iris)

# Hypothesis testing
t_test <- t.test(Sepal.Length ~ Species, data = iris %>%
                    filter(Species %in% c("setosa", "versicolor")))
print(t_test)

# Correlation
cor_test <- cor.test(iris$Sepal.Length, iris$Petal.Length)
print(cor_test)

# Normality check
shapiro_test <- shapiro.test(iris$Sepal.Length)
print(shapiro_test)

# Binary classification: setosa vs non-setosa
model_data <- iris %>%
  mutate(target = ifelse(Species == "setosa", 1, 0)) %>%
  select(Sepal.Length, Sepal.Width, Petal.Length, Petal.Width, target)

set.seed(42)
train_idx <- createDataPartition(model_data$target, p = .80, list = FALSE)
train <- model_data[train_idx, ]
test <- model_data[-train_idx, ]

model <- glm(target ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width,
             data = train, family = binomial())

prob <- predict(model, newdata = test, type = "response")
pred <- ifelse(prob >= .5, 1, 0)

cm <- confusionMatrix(
  factor(pred, levels = c(0,1)),
  factor(test$target, levels = c(0,1)),
  positive = "1"
)
print(cm)

roc_obj <- roc(test$target, prob)
print(auc(roc_obj))

# 5-fold cross-validation
ctrl <- trainControl(method = "cv", number = 5,
                     classProbs = TRUE,
                     summaryFunction = twoClassSummary)

cv_data <- iris %>%
  mutate(target = factor(ifelse(Species == "setosa", "setosa", "other"),
                         levels = c("setosa", "other"))) %>%
  select(Sepal.Length, Sepal.Width, Petal.Length, Petal.Width, target)

set.seed(42)
cv_model <- train(target ~ ., data = cv_data,
                  method = "glm", family = "binomial",
                  metric = "ROC", trControl = ctrl)
print(cv_model)

# ============================================================
# WEEK 4: Reporting
# ============================================================
# Recommended workflow:
# 1. Validate data quality before analysis.
# 2. Use visualization to discover patterns.
# 3. Use statistical tests to assess evidence.
# 4. Build and validate a predictive model.
# 5. Translate results into clear, non-technical recommendations.
