# Predicting Patient Readmissions Using Logistic Regression and Random Forest

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python) 
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-0.25-orange?logo=scikit-learn) 
![Pandas](https://img.shields.io/badge/Pandas-1.5-brightgreen?logo=pandas) 
![NumPy](https://img.shields.io/badge/NumPy-1.25-yellow?logo=numpy)

---

## Project Overview
Hospital readmissions are costly and can negatively impact patient health. This project predicts which patients are at risk of readmission using demographic and clinical data such as **age, gender, blood pressure, and cholesterol**. Predictive models including **Logistic Regression** and **Random Forest** were applied to identify high-risk patients and support early intervention.

---

## Objective
- Predict patient readmissions using historical demographic and clinical data.
- Enable healthcare providers to intervene early and reduce unnecessary hospital readmissions.

---

## Approach

### 1. Exploratory Data Analysis (EDA)
- Checked for **missing values, outliers, and imbalanced classes**.
- Visualized distributions of **age, blood pressure, and cholesterol**.
- Examined correlations between features and readmission outcome to guide feature selection.

### 2. Data Preprocessing
- Parsed **blood pressure** into systolic and diastolic numeric values.
- Encoded **gender** as numeric values for modeling.
- Normalized numeric features (**age, systolic, diastolic, cholesterol**) for better model performance.
- Addressed missing values using **imputation** where necessary.

### 3. Modeling
- Built **Logistic Regression** and **Random Forest** classifiers.
- Handled class imbalance using **SMOTE** and **class weighting** to improve detection of high-risk patients.

---

## Evaluation

### Logistic Regression Report
          precision    recall  f1-score   support
       0       0.89      0.58      0.70      5231
       1       0.15      0.50      0.23       769
accuracy                           0.57      6000


### Random Forest Report
          precision    recall  f1-score   support
       0       0.87      0.99      0.93      5231
       1       0.16      0.01      0.02       769
accuracy                           0.87      6000

## Insights
- **Logistic Regression** is better at detecting high-risk patients (recall for readmitted = 0.50).  
- **Random Forest** favors the majority class and misses almost all readmissions (recall for readmitted = 0.01).  
- For healthcare applications, **recall for the readmitted class is critical**, so Logistic Regression is preferred.  
- Both models show **low precision for readmitted patients**, suggesting further feature engineering or alternative models could improve performance.

---

## Outcome & Impact
- Logistic Regression successfully identifies patients at risk of readmission.  
- Insights can help healthcare providers:
  - Intervene early.
  - Improve patient outcomes.
  - Reduce unnecessary hospital readmissions.  
- Demonstrates the ability to translate patient data into **actionable insights** using predictive analytics.

---

## Tools & Skills
- **Programming & Libraries:** Python, Pandas, NumPy, Scikit-learn  
- **Data Analysis:** EDA, feature engineering, data preprocessing  
- **Machine Learning:** Logistic Regression, Random Forest, handling imbalanced data (SMOTE, class weighting)  
- **Evaluation & Visualization:** Accuracy, Precision, Recall, F1-score, Matplotlib, Seaborn
