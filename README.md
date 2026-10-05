# Statistical Analysis and Predictive Modeling using R — Week 3

A reproducible R project for statistical analysis, hypothesis testing, classification modeling, diagnostics, and cross-validation using the publicly available **iris** dataset.

## Project Objective

The objective is to demonstrate a complete statistical and predictive modeling workflow in R:

- Explore the selected dataset and variables
- Perform hypothesis testing
- Check correlation and distribution assumptions
- Build a binary logistic regression model
- Split the data into training and testing sets
- Evaluate classification performance
- Use cross-validation to assess model stability
- Present diagnostics and analytical insights

## Dataset

The project uses the classic `iris` dataset available in R. The target is converted into a binary classification problem:

- `Setosa` = 1
- `Non-setosa` = 0

The main predictors are the four flower measurements: Sepal Length, Sepal Width, Petal Length, and Petal Width.

## Methods Used

1. Descriptive statistics
2. Welch two-sample t-test
3. Pearson correlation test
4. Shapiro-Wilk normality test
5. Logistic regression
6. Train/test split
7. Confusion matrix
8. Accuracy, precision, recall and F1-score
9. ROC curve and AUC
10. Five-fold cross-validation
11. Residual/model diagnostic checks

## Requirements

Install R and the following packages:

```r
install.packages(c("dplyr", "caret", "pROC"))
```

## How to Run

Open `week3_statistical_predictive_modeling.R` in RStudio and run the script from top to bottom.

The script is written as a single reproducible workflow and uses `set.seed(42)` so that the train/test split and cross-validation are reproducible.

## Repository Contents

- `week3_statistical_predictive_modeling.R` — complete R analysis and modeling code
- `Week_3_Statistical_Analysis_Predictive_Modeling_Report.docx` — final internship report
- `outputs/` — generated charts and model evaluation visuals

## Key Learning Outcome

The project demonstrates how statistical testing and predictive modeling can be combined to move from exploratory analysis to an evaluated machine-learning model. The emphasis is on choosing appropriate methods, interpreting results clearly, and validating model performance rather than relying on a single accuracy value.

## Author

**Shivam Nakoti**
