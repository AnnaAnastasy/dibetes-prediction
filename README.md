# Diabetes Prediction: From Logistic Regression to Advanced Ensemble Methods

This project explores diabetes prediction using various machine learning techniques, starting with logistic regression and advancing to powerful ensemble methods like Random Forests and Gradient Boosting.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
4. [Machine Learning Models](#machine-learning-models)
5. [Evaluation](#evaluation)
6. [Conclusions](#conclusions)
7. [How to Run the Notebook](#how-to-run-the-notebook)

---

## 1. Project Overview

Diabetes is a prevalent health concern globally, and early prediction can significantly improve patient outcomes. This project uses machine learning to predict whether an individual has diabetes based on clinical variables.

### Key Objectives:
- Compare the performance of different machine learning models.
- Analyze feature importance in predicting diabetes outcomes.
- Highlight the advantages of advanced ensemble techniques over simpler methods.

---

## 2. Dataset

The dataset used for this project contains medical details such as glucose levels, BMI, and insulin measurements.

### Key Information:
- **Source:** [Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/kandij/diabetes-dataset)
- **Size:** 768 rows, 9 numerical columns.
- **Target Variable:** `Outcome` (1 = Diabetes, 0 = No Diabetes).

---

## 3. Exploratory Data Analysis (EDA)

- Identified missing or anomalous values (e.g., zeros in glucose and insulin levels).
- Examined distributions of key features like BMI, Age, and Glucose.
- Explored relationships between features and the `Outcome` variable.

---

## 4. Machine Learning Models

This project implements and compares the following models:
1. **Logistic Regression** (Baseline model)
2. **Decision Trees**
3. **Random Forest**
4. **Gradient Boosting (XGBoost, LightGBM, etc.)**

For each model, hyperparameter tuning was performed to optimize performance.

---

## 5. Evaluation

Models were evaluated based on:
- **Accuracy**
- **Precision, Recall, and F1-Score**
- **ROC-AUC Score**

Advanced ensemble models outperformed logistic regression and simple decision trees, demonstrating their ability to handle complex relationships and interactions between features.

---

## 6. Conclusions

- Ensemble models like Random Forest and Gradient Boosting provide better performance for predicting diabetes compared to simpler models.
- Feature importance analysis highlights glucose levels and BMI as key predictors.
- Proper handling of missing or invalid data (e.g., replacing zeros) is crucial for building reliable models.

---

## 7. How to Run the Notebook

### Prerequisites
- Python 3.8 or higher
- Libraries: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`, `lightgbm`.

### Setup
1. Install required libraries:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm
