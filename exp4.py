# ------------------------------------------------------------
# Experiment: AI for Medical Diagnosis (Heart Disease Prediction)
# ------------------------------------------------------------

# -------------------------
# 1. Import Libraries
# -------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ML Algorithms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Evaluation Metrics
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -------------------------
# 2. Load Dataset
# -------------------------
df = pd.read_csv("heart.csv")  # Ensure 'heart.csv' is in your working directory
print("✅ Dataset Loaded Successfully!\n")
print(df.head())

# -------------------------
# 3. Initial Exploration
# -------------------------
print("\n--- Dataset Info ---")
print(df.info())

print("\n--- Correlation Matrix ---")
print(df.corr())

# -------------------------
# 4. Data Preprocessing
# -------------------------

# Drop less correlated columns if present
drop_cols = ['oldpeak', 'slp', 'thall']
df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

# Check for null values
print("\n--- Checking for Null Values ---")
print(df.isnull().sum())

# -------------------------
# 5. Exploratory Data Analysis (EDA)
# -------------------------
plt.figure(figsize=(12, 6))
sns.countplot(x='age', data=df, palette='coolwarm')
plt.title('Age Distribution of Patients')
plt.xticks(rotation=90)
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x='sex', data=df, palette='Set2')
plt.title('Gender Distribution (0: Female, 1: Male)')
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x='cp', data=df, palette='Set3')
plt.title('Chest Pain Type Distribution')
plt.show()

plt.figure(figsize=(6,4))
sns.histplot(df['trtbps'], bins=20, kde=True, color='skyblue')
plt.title('Resting Blood Pressure Distribution')
plt.show()

plt.figure(figsize=(6,4))
sns.histplot(df['chol'], bins=20, kde=True, color='lightgreen')
plt.title('Cholesterol Level Distribution')
plt.show()

plt.figure(figsize=(6,4))
sns.histplot(df['thalachh'], bins=20, kde=True, color='salmon')
plt.title('Maximum Heart Rate Distribution')
plt.show()

# -------------------------
# 6. Feature Scaling and Splitting
# -------------------------
X = df.drop('output', axis=1)  # Features
y = df['output']               # Target

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
print("\n✅ Data split into Train and Test sets successfully.")

# Encode target if necessary
encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.transform(y_test)

# -------------------------
# 7. Model Training and Evaluation
# -------------------------
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(kernel='linear', random_state=42)
}

accuracy_results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    accuracy_results.append({"Model": name, "Accuracy": acc})
    
    print(f"\n--- {name} ---")
    print(f"Accuracy: {acc:.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

# -------------------------
# 8. Model Comparison
# -------------------------
accuracy_df = pd.DataFrame(accuracy_results)
accuracy_df = accuracy_df.sort_values(by='Accuracy', ascending=False)

print("\n--- Model Performance Summary ---")
print(accuracy_df)

plt.figure(figsize=(8,4))
sns.barplot(x='Model', y='Accuracy', data=accuracy_df, palette='mako')
plt.title('Model Accuracy Comparison')
plt.xticks(rotation=30)
plt.show()

# -------------------------
# 9. Conclusion
# -------------------------
best_model = accuracy_df.iloc[0]
print(f"\n🏆 Best Performing Model: {best_model['Model']} with Accuracy: {best_model['Accuracy']:.4f}")
