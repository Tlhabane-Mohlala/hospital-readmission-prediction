# Predicting Patient Readmissions Using Logistic Regression and Random Forest

## Project Overview
Hospital readmissions are costly and can negatively impact patient health. This project focuses on predicting which patients are at risk of readmission using demographic and clinical data such as **age, gender, blood pressure, and cholesterol**. Predictive models like **Logistic Regression** and **Random Forest** were applied to identify high-risk patients and support early intervention.

## Objective
- Predict patient readmissions using historical demographic and clinical data.
- Enable healthcare providers to intervene early and reduce unnecessary hospital readmissions.

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

### 4. Evaluation
- Evaluated models using **accuracy, precision, recall, and F1-score**.
- Prioritized **recall** to ensure high-risk patients were correctly identified.


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

## How to Run
1. Clone this repository:  
```bash
git clone https://github.com/Tlhabane-Mohlala/patient-readmission-prediction.git

