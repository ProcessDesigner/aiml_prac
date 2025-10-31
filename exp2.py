# ===========================================
# 📘 STEP 1: Import Libraries
# ===========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-whitegrid')

# Load cleaned dataset
df = pd.read_csv('healthcare_diabetes_cleaned.csv')

# ===========================================
# 📗 STEP 2: Basic Overview
# ===========================================
print("🔹 Dataset Dimensions:", df.shape)
print("🔹 Columns:", df.columns.tolist(), "\n")

print("🔹 First 5 Rows:")
print(df.head(), "\n")

print("🔹 Dataset Info:")
print(df.info(), "\n")

print("🔹 Statistical Summary:")
print(df.describe(), "\n")

# ===========================================
# 📘 STEP 3: Check Missing & Duplicate Data
# ===========================================
print("🔹 Missing Values per Column:")
print(df.isnull().sum(), "\n")

print("🔹 Duplicate Records:", df.duplicated().sum(), "\n")

# Visualize missing data
plt.figure(figsize=(10,5))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title("🩺 Missing Values Heatmap")
plt.show()

# ===========================================
# 📊 STEP 4: Univariate Analysis
# ===========================================
# Analyze each variable individually

# Numeric column distribution
num_cols = df.select_dtypes(include=np.number).columns

for col in num_cols:
    plt.figure(figsize=(7,4))
    sns.histplot(df[col], kde=True, bins=30, color='teal')
    plt.title(f"📊 Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show()

# Boxplot for outliers
for col in num_cols:
    plt.figure(figsize=(6,3))
    sns.boxplot(df[col], color='lightcoral')
    plt.title(f"📦 Outlier Detection for {col}")
    plt.show()

# ===========================================
# 📈 STEP 5: Bivariate Analysis
# ===========================================
# Example: Relationship between features and target (Outcome)

if 'Outcome' in df.columns:
    target = 'Outcome'
    for col in num_cols:
        if col != target:
            plt.figure(figsize=(7,4))
            sns.boxplot(x=target, y=col, data=df, palette='coolwarm')
            plt.title(f"🔹 {col} vs {target}")
            plt.show()

# ===========================================
# 💠 STEP 6: Correlation Analysis
# ===========================================
plt.figure(figsize=(10,7))
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("💠 Correlation Heatmap")
plt.show()

# Show top correlated features with target
if 'Outcome' in df.columns:
    print("🔹 Top Correlations with Outcome:")
    print(corr['Outcome'].sort_values(ascending=False), "\n")

# ===========================================
# 📊 STEP 7: Pairplot - Relationship Between All Features
# ===========================================
# (If too many columns, select top ones)
sns.pairplot(df.sample(min(200, len(df))), diag_kind='kde')
plt.suptitle("🧩 Pairplot of Features", y=1.02)
plt.show()

# ===========================================
# 🔍 STEP 8: Categorical Feature Analysis (if any)
# ===========================================
cat_cols = df.select_dtypes(exclude=np.number).columns

for col in cat_cols:
    plt.figure(figsize=(6,3))
    sns.countplot(x=col, data=df, palette='pastel')
    plt.title(f"📊 Count of {col}")
    plt.show()

# ===========================================
# 🧮 STEP 9: Target Variable Distribution
# ===========================================
if 'Outcome' in df.columns:
    plt.figure(figsize=(6,4))
    sns.countplot(x='Outcome', data=df, palette='Set2')
    plt.title("🩸 Distribution of Outcome (Diabetes vs Non-Diabetes)")
    plt.xlabel("Outcome (0 = No Diabetes, 1 = Diabetes)")
    plt.ylabel("Count")
    plt.show()

    print("🔹 Outcome Value Counts:")
    print(df['Outcome'].value_counts(), "\n")

# ===========================================
# 📈 STEP 10: Relationship Between Key Medical Features
# ===========================================
# Example features in diabetes dataset (modify as per your dataset)
features_to_compare = ['Glucose', 'Insulin', 'BMI', 'Age']

sns.pairplot(df[features_to_compare + ['Outcome']], hue='Outcome', palette='husl')
plt.suptitle("📈 Relationship Between Key Features and Outcome", y=1.02)
plt.show()

# ===========================================
# 📉 STEP 11: Skewness and Kurtosis
# ===========================================
print("🔹 Skewness:")
print(df.skew(), "\n")
print("🔹 Kurtosis:")
print(df.kurt(), "\n")

# ===========================================
# 📊 STEP 12: Feature Importance (Optional - Correlation based)
# ===========================================
if 'Outcome' in df.columns:
    corr_target = abs(corr['Outcome']).sort_values(ascending=False)
    plt.figure(figsize=(8,4))
    sns.barplot(x=corr_target.index, y=corr_target.values, palette='rocket')
    plt.title("💡 Feature Importance (Correlation with Outcome)")
    plt.xticks(rotation=45)
    plt.ylabel("Correlation Strength")
    plt.show()

# ===========================================
# ✅ STEP 13: Save EDA Report Summary
# ===========================================
summary = {
    'Total_Rows': df.shape[0],
    'Total_Columns': df.shape[1],
    'Missing_Values': df.isnull().sum().sum(),
    'Duplicates': df.duplicated().sum(),
    'Numerical_Columns': len(num_cols),
    'Categorical_Columns': len(cat_cols)
}

summary_df = pd.DataFrame([summary])
summary_df.to_csv('EDA_summary.csv', index=False)
print("💾 EDA Summary saved as 'EDA_summary.csv'.")
