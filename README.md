## Census Income Data Classification

### Project Overview

This project aims to classify whether an individual's income is **greater than $50,000 or less than or equal to $50,000**. Using demographic and socioeconomic data from the 1994 US Census, the goal is to develop a machine learning model that can predict income brackets. This type of analysis is valuable for understanding economic disparities and social mobility.

-----

### Technical Highlights

  * **Dataset**: [Kaggle - US Adult Income Update](https://www.kaggle.com/datasets/vivamoto/us-adult-income-update)
  * **Size**: 48842 entries, 15 columns. After data cleaning (dropping duplicates) and balancing, the dataset size for training and testing is adjusted.
  * **Key Features**:
      * `age`, `workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `capital-gain`, `capital-loss`, `hours-per-week`, `native-country`.
  * **Approach**:
      * **Data Cleaning**: Dropped duplicate rows. The dataset initially contains '?' as a placeholder for missing values, which are handled implicitly by the subsequent label encoding step.
      * **Exploratory Data Analysis**: Histograms, Boxplots, and a heatmap were used for visualization to understand data distributions and correlations.
      * **Label Encoding**: Applied to all columns, including categorical features and the target 'income'.
      * **Handling Class Imbalance**: The original dataset is imbalanced (`<=50K`: 37155, `>50K`: 11687), so `SMOTE` was applied to the training data to balance the classes before training.
      * **Binary Classification**: The target variable `income` is classified into two categories: `<=50K` and `>50K`.
      * **Models Used**:
          * Logistic Regression, Ridge Classifier, SVC, Random Forest, XGBoost, AdaBoost, Gradient Boosting, Bagging, Decision Tree.
  * **Best Accuracy**:
      * 90.3% with XGBoost Classifier.
      * 89.5% with Random Forest Classifier.
      * 88.5% with Bagging Classifier.

-----

### Purpose and Applications

  * **Predict income brackets** of individuals based on their demographic and employment data.
  * Assist economic researchers and policymakers in analyzing factors that contribute to income levels and economic inequality.
  * Provide a tool for targeted marketing or resource allocation by businesses and non-profits.
  * Demonstrate how machine learning can be used to understand and model complex social phenomena.

-----

### Installation

Clone the repository:

```bash
git clone https://github.com/BhaveshBhakta/Census-Income-Data-Classification.git
cd Census-Income-Data-Classification
```

Install the necessary libraries:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn imbalanced-learn xgboost
```

-----

### Collaboration

We welcome contributions to improve the project. You can help by:

  * Performing comprehensive hyperparameter tuning and cross-validation for all models to achieve optimal performance.
  * Exploring more robust data imputation methods for the '?' values instead of implicit encoding.
  * Investigating the impact of different feature engineering strategies.
  * Adding explainability (e.g., SHAP or LIME) to understand which features are the most significant predictors of income
