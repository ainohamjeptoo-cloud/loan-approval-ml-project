# Machine Learning Model for Loan Approval

## Project Overview

FinTech Innovations is a growing financial technology company seeking to modernize its loan approval process. The current process relies heavily on manual review, which can result in inconsistent decisions and slower responses to applicants.

This project develops a **machine learning classification model** to predict whether a historical loan application was classified as approved or not approved based on financial, credit, employment, and behavioral characteristics.

The project follows the **CRISP-DM (Cross-Industry Standard Process for Data Mining)** framework, covering business understanding, data understanding, data preparation, modeling, evaluation, and business recommendations.

---

## Business Problem

Loan approval errors can have different financial consequences:

* Denying a creditworthy applicant may result in approximately **$8,000 in lost profit**.
* Approving an applicant whose loan subsequently defaults may result in approximately **$50,000 in losses**.

Because these costs are asymmetric, model performance cannot be evaluated using accuracy alone. The project therefore considers multiple classification metrics, including:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC

The analysis also considers the potential financial impact of false-positive and false-negative predictions.

---

## Dataset

The dataset contains **20,000 historical loan applications**.

The target variable is:

* `LoanApproved` — whether the historical application was approved (`1`) or not approved (`0`).

The dataset contains several types of applicant information, including:

### Financial Features

* Annual income
* Monthly income
* Loan amount
* Monthly loan payment
* Savings account balance
* Checking account balance
* Total assets
* Total liabilities
* Net worth
* Debt-to-income ratios
* Interest rates

### Credit Features

* Credit score
* Risk score
* Number of credit inquiries
* Number of open credit lines
* Previous loan defaults
* Bankruptcy history

### Applicant Characteristics

* Age
* Education level
* Employment status
* Job tenure
* Marital status
* Home ownership
* Number of dependents
* Loan purpose

---

## Methodology

The project follows the CRISP-DM framework.

### 1. Business Understanding

The business problem, stakeholders, financial consequences of prediction errors, and appropriate evaluation metrics were identified.

### 2. Data Understanding

The dataset was inspected for:

* Data types
* Missing values
* Duplicate records
* Target distribution
* Categorical values
* Descriptive statistics
* Outliers
* Feature relationships
* Correlations

Exploratory data analysis included boxplots, a correlation matrix, and analysis of loan approval outcomes and important financial and risk variables.

### 3. Feature Engineering

Additional features were created to capture relationships between financial variables, including:

* `LoanToIncomeRatio`
* `AssetToLiabilityRatio`

These features provide additional information about an applicant's financial position.

### 4. Data Preparation

A Scikit-learn preprocessing pipeline was developed using `ColumnTransformer`.

The preprocessing workflow includes:

* Median imputation for numerical variables
* Most-frequent imputation for categorical variables
* One-hot encoding of categorical variables
* Standardization of numerical variables
* Separate preprocessing pipelines for numerical and categorical features

A stratified train-test split was used to preserve the target-class distribution.

### 5. Model Development

Four classification algorithms were evaluated:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Gradient Boosting

Five-fold stratified cross-validation was used to compare model performance.

### 6. Hyperparameter Tuning

`GridSearchCV` with five-fold cross-validation was used to optimize the models.

The primary tuning metric was **F1-score**, because it balances precision and recall and is useful when both types of classification errors matter.

### 7. Model Evaluation

The final model was evaluated on an independent test set using:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Confusion matrix
* ROC curve

Feature coefficients were also examined to understand which variables had the strongest influence on the Logistic Regression model.

---

## Model Results

The tuned Logistic Regression model achieved the following performance on the test set:

| Metric    | Score |
| --------- | ----: |
| Accuracy  | 1.000 |
| Precision | 0.999 |
| Recall    | 1.000 |
| F1-score  | 0.999 |
| ROC-AUC   | 1.000 |

The test-set confusion matrix contained:

* True Negatives: **3,043**
* False Positives: **1**
* False Negatives: **0**
* True Positives: **956**

These results indicate extremely high predictive performance on the available test data.

However, the results should be interpreted cautiously because the `RiskScore` variable shows very strong separation between approved and non-approved applications. The timing and definition of this variable should be confirmed before deployment to ensure that it was available at the time of the original loan decision and does not introduce data leakage.

---

## Business Impact

Using the provided business cost assumptions:

* False positive cost: **$50,000**
* False negative cost: **$8,000**

The test set contained:

* 1 false positive
* 0 false negatives

The corresponding potential error cost is:

**$50,000**

This should not be interpreted as an actual financial loss. The model predicts historical loan approval outcomes rather than whether an approved loan will ultimately default. The $50,000 represents a potential cost under the assignment's assumption if the false-positive case subsequently resulted in a default.

---

## Key Limitations

Several limitations should be considered before using the model in a real-world loan approval environment.

### Potential Data Leakage

`RiskScore` shows a strong relationship with the target variable. Its definition and timing should be investigated to determine whether it was available before the loan approval decision.

### Historical Bias

The model learns from historical approval decisions. If historical decisions contained biases or inconsistencies, the model may reproduce those patterns.

### Generalization

The model was evaluated using historical data. Performance on future applicants may differ if applicant behavior, economic conditions, lending policies, or data distributions change.

### Business Outcome

Predicting loan approval is different from predicting loan default. A separate default-risk model may be necessary to directly estimate the probability of repayment failure.

### Human Oversight

The model should support, rather than completely replace, appropriate human review and risk-management procedures.

---

## Recommendations

Based on the analysis, the following steps are recommended before deployment:

1. Validate the definition and timing of `RiskScore`.
2. Test the model on newer and independent data.
3. Monitor precision, recall, F1-score, and ROC-AUC over time.
4. Monitor financial outcomes rather than relying only on classification metrics.
5. Investigate potential bias across relevant applicant groups.
6. Maintain appropriate human oversight for high-risk decisions.
7. Consider developing a separate model specifically for loan default risk.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook
* Git
* GitHub

---

## Project Structure

```text
Loan_Approval_Project/
│
├── financial_loan_risk.ipynb
├── financial_loan_data.csv
└── README.md
```

---

## Conclusion

This project demonstrates an end-to-end machine learning workflow for a loan approval classification problem. The analysis covers business understanding, exploratory data analysis, feature engineering, preprocessing, model comparison, hyperparameter tuning, evaluation, and business impact.

The Logistic Regression model achieved near-perfect performance on the available test set. However, the unusually high performance highlights the importance of investigating potential data leakage, particularly involving the `RiskScore` variable, before considering real-world deployment.

The project demonstrates how machine learning performance can be evaluated alongside financial costs, model limitations, interpretability, and practical business considerations.
