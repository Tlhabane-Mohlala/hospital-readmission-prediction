# Install libraries
#%pip install imbalanced-learn seaborn
#%matplotlib inline

# Import the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# -----------------------------
# 1. LOAD DATA (SPARK → PANDAS)
# -----------------------------
df_spark = spark.table("default.hospital_readmissions_30_k")

# Convert to pandas
df = df_spark.toPandas()

# ======================================================
# EDA SECTION
# ======================================================

# -----------------------------
# 1. Preview & Missing Values
# -----------------------------
print("\n----- DATA HEAD -----")
print(df.head())

print("\n----- DATA INFO -----")
print(df.info())

print("\n----- SUMMARY STATISTICS -----")
print(df.describe())

print("\n----- MISSING VALUES -----")
print(df.isnull().sum())

# Heatmap of missing values
plt.figure(figsize=(8,4))
sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
plt.title("Missing Values Heatmap")
plt.show()
# -----------------------------
# Target distribution
# -----------------------------
plt.figure(figsize=(6,4))
ax = sns.countplot(data=df, x='readmitted_30_days')

# Add count labels on each bar
for container in ax.containers:
    ax.bar_label(container)

plt.title("Target Distribution")
plt.xlabel("Readmitted in 30 Days")
plt.ylabel("Count")
plt.show()

# -----------------------------
# 2. Parse Blood Pressure
# -----------------------------
df[['systolic', 'diastolic']] = df['blood_pressure'].str.split('/', expand=True).astype(int)
df.drop(columns=['blood_pressure'], inplace=True)

# -----------------------------
# 3. Encode categorical variables
# -----------------------------
cat_cols = ['gender', 'diabetes', 'hypertension', 'discharge_destination']
for col in cat_cols:
    df[col] = LabelEncoder().fit_transform(df[col])

df['readmitted_30_days'] = df['readmitted_30_days'].map({'Yes': 1, 'No': 0})

# -----------------------------
# 4. Define X and y
# -----------------------------
X = df.drop(columns=['patient_id', 'readmitted_30_days'])
y = df['readmitted_30_days']

# -----------------------------
# 5. Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 6. Scaling numerical features
# -----------------------------
num_cols = ['age', 'cholesterol', 'bmi', 'medication_count',
            'length_of_stay', 'systolic', 'diastolic']

scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# -----------------------------
# 7. SMOTE Oversampling
# -----------------------------
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# -----------------------------
# 8. LOGISTIC REGRESSION
# -----------------------------
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_res, y_train_res)
y_pred_log = log_model.predict(X_test)

print("Logistic Regression Report:")
print(classification_report(y_test, y_pred_log))

# -----------------------------
# 9. RANDOM FOREST
# -----------------------------
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_res, y_train_res)
y_pred_rf = rf_model.predict(X_test)

print("Random Forest Report:")
print(classification_report(y_test, y_pred_rf))

# -----------------------------
# 10. CONFUSION MATRIX PLOTS
# -----------------------------
def plot_cm(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

plot_cm(y_test, y_pred_log, "Logistic Regression Confusion Matrix")
plot_cm(y_test, y_pred_rf, "Random Forest Confusion Matrix")

# -----------------------------
# 11. FEATURE IMPORTANCE
# -----------------------------
importances = rf_model.feature_importances_
features = X_train.columns

feat_importance = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

sns.barplot(data=feat_importance, x='Importance', y='Feature')
plt.title("Random Forest Feature Importance")
plt.show()
