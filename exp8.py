# ===========================
# Experiment: Diabetes Prediction using ML
# ===========================

# Step 1: Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ===========================
# Step 2: Load Dataset
# ===========================
# Load Pima Indians Diabetes dataset (ensure it's in your working directory)
df = pd.read_csv('diabetes.csv')

# Display first few rows
print("🔹 First 5 Rows of Data:")
print(df.head())

# ===========================
# Step 3: Basic Data Information
# ===========================
print("\n🔹 Dataset Info:")
print(df.info())

print("\n🔹 Summary Statistics:")
print(df.describe())

print("\n🔹 Missing Values per Column:")
print(df.isnull().sum())

# ===========================
# Step 4: Data Cleaning
# ===========================
# Replace 0 values with NaN for certain columns (not possible in reality)
cols_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_with_zero] = df[cols_with_zero].replace(0, np.nan)

# Fill missing values with column mean
df.fillna(df.mean(), inplace=True)

# ===========================
# Step 5: Exploratory Data Analysis (EDA)
# ===========================

# Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Features')
plt.show()

# Distribution of Outcome
plt.figure(figsize=(5,4))
sns.countplot(x='Outcome', data=df, palette='Set2')
plt.title('Diabetes Outcome Distribution (0 = No, 1 = Yes)')
plt.show()

# Pairplot for selected features
sns.pairplot(df[['Glucose', 'BMI', 'Age', 'Insulin', 'Outcome']], hue='Outcome', palette='husl')
plt.show()

# ===========================
# Step 6: Feature Selection and Splitting
# ===========================
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ===========================
# Step 7: Feature Scaling
# ===========================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ===========================
# Step 8: Model Building
# ===========================
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ===========================
# Step 9: Model Evaluation
# ===========================
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"\n✅ Model Accuracy: {accuracy * 100:.2f}%")
print("\n🔹 Confusion Matrix:")
print(cm)
print("\n🔹 Classification Report:")
print(classification_report(y_test, y_pred))

# ===========================
# Step 10: Feature Importance
# ===========================
feature_importance = pd.Series(model.feature_importances_, index=df.columns[:-1])
feature_importance = feature_importance.sort_values(ascending=False)

plt.figure(figsize=(8,5))
sns.barplot(x=feature_importance, y=feature_importance.index, palette='viridis')
plt.title("Feature Importance in Diabetes Prediction")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.show()

# ===========================
# Step 11: Save Model (Optional)
# ===========================
import joblib
joblib.dump(model, 'diabetes_prediction_model.pkl')
print("\n💾 Model saved successfully as 'diabetes_prediction_model.pkl'")
