import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns

df = pd.read_csv('health care diabetes.csv')

print('head value',df.head())

print('dataset info',df.info())

print('Null values',df.isnull().sum())

print('unique value in all columns ',df.nunique())


# If numeric columns have missing values, fill them with mean
num_cols = df.select_dtypes(include=[np.number]).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

# If categorical columns have missing values, fill them with mode
cat_cols = df.select_dtypes(exclude=[np.number]).columns
for col in cat_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)


for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[col] = np.where(df[col] < lower, lower, df[col])
    df[col] = np.where(df[col] > upper, upper, df[col])

print("✅ Outliers handled using IQR method.\n")

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

print("✅ Categorical columns encoded successfully.\n")


# =============================
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

print("✅ Numeric columns normalized to [0,1] range.\n")

# Example: Calculate BMI if 'Weight' and 'Height' exist
if 'Weight' in df.columns and 'Height' in df.columns:
    df['BMI'] = df['Weight'] / (df['Height']/100)**2
    print("✅ New feature 'BMI' created.")

if 'Age' in df.columns:
    df['Age_Group'] = pd.cut(df['Age'], bins=[0,30,50,80], labels=['Young','Middle','Old'])

from sklearn.utils import resample

majority = df[df['Outcome'] == 0]
minority = df[df['Outcome'] == 1]

minority_upsampled = resample(minority, 
                              replace=True,
                              n_samples=len(majority),
                              random_state=42)
df = pd.concat([majority, minority_upsampled])
print("✅ Dataset balanced using upsampling.\n")

skewed_cols = ['Glucose', 'Insulin']
for col in skewed_cols:
    if col in df.columns:
        df[col] = np.log1p(df[col])
        print(f"✅ Log transformation applied to {col}.")

df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
print("✅ One-hot encoding applied for categorical columns.")

corr = df.corr()
high_corr = [col for col in corr.columns if any(abs(corr[col]) > 0.9)]
print("Highly correlated columns:", high_corr)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# =============================
# 📘 Step 9: Data Integration (if applicable)
# =============================
# If you have another dataset (e.g., patient demographics, hospital records),
# you can merge them using a common key like 'Patient_ID'.
# Example (commented out):
# df = df.merge(other_df, on='Patient_ID', how='left')

print("🔹 Cleaned & Transformed Data Overview:")
print(df.head(), "\n")
print(df.describe(), "\n")
print(df.info(), "\n")

# =============================
# 📊 Step 11: Save Cleaned Data
# =============================
df.to_csv('healthcare_diabetes_cleaned.csv', index=False)
print("💾 Cleaned data saved as 'healthcare_diabetes_cleaned.csv'.")

# print(df.iloc[0:,1:4])

