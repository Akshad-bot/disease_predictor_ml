"""
scripts/build_notebook_01.py
Constructs and executes notebooks/01_eda_preprocessing.ipynb
"""

import nbformat as nbf
from nbclient import NotebookClient

def create_eda_notebook():
    nb = nbf.v4.new_notebook()
    cells = []

    # Title & Metadata
    cells.append(nbf.v4.new_markdown_cell("""# Multi-Disease Risk Prediction System
## Notebook 01: Exploratory Data Analysis & Preprocessing Pipeline
**Academic Coursework:** Machine Learning Project  
**Syllabus Coverage:** **UNIT I & UNIT II**  
- **UNIT I:** Data Understanding, Data Collection, Cleaning, Handling Missing Values, Inspection  
- **UNIT II:** Exploratory Data Analysis (EDA) - Univariate, Bivariate, Multivariate, Dimensionality Reduction (PCA), and Scikit-Learn Preprocessing Pipelines

---

### Overview of Diseases Analyzed:
1. **Diabetes:** Pima Indians Diabetes Dataset (`data/diabetes.csv`)
2. **Heart Disease:** Comprehensive Heart Disease Dataset (`data/heart.csv`)
3. **Chronic Kidney Disease (CKD):** Clinical Kidney Dataset (`data/kidney_disease.csv`)
4. **Breast Cancer:** Diagnostic Wisconsin Dataset (`sklearn.datasets.load_breast_cancer`)
"""))

    # Imports Cell
    cells.append(nbf.v4.new_code_cell("""# Environment & Library Imports
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

# Configure calm visualization aesthetics
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#E5E7EB'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.color'] = '#F3F4F6'
PRIMARY_COLOR = '#0F766E'
ACCENT_COLOR = '#D97706'
DANGER_COLOR = '#DC2626'
PALETTE = ['#0F766E', '#DC2626']

print("Libraries imported successfully!")
"""))

    # UNIT I: Diabetes
    cells.append(nbf.v4.new_markdown_cell("""---
## 1. Diabetes Dataset: Inspection & Preprocessing (UNIT I)
**Target:** `Outcome` (0 = Non-Diabetic, 1 = Diabetic)  
**Clinical Rule:** Biological parameters such as `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI` cannot physically be zero in living individuals. Zero values represent unrecorded/missing measurements and must be imputed.
"""))

    cells.append(nbf.v4.new_code_cell("""# 1.1 Load and Inspect Diabetes Data
df_diab = pd.read_csv('../data/diabetes.csv')
print("Diabetes Dataset Shape:", df_diab.shape)
print("\\n--- First 5 Rows ---")
display(df_diab.head())

print("\\n--- DataFrame Info ---")
df_diab.info()

print("\\n--- Null Values Count ---")
print(df_diab.isna().sum())

print("\\n--- Zero Values Count (Potential Missing) ---")
zero_counts = (df_diab == 0).sum()
print(zero_counts[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']])
"""))

    cells.append(nbf.v4.new_code_cell("""# 1.2 Data Quality Fix: Replace Zeros with NaN in Specific Columns
zero_impute_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df_diab_cleaned = df_diab.copy()
df_diab_cleaned[zero_impute_cols] = df_diab_cleaned[zero_impute_cols].replace(0, np.nan)

print("Missing values after converting zeros to NaN:")
print(df_diab_cleaned[zero_impute_cols].isna().sum())
"""))

    # UNIT II: Diabetes EDA
    cells.append(nbf.v4.new_markdown_cell("""### 1.3 Diabetes Exploratory Data Analysis (UNIT II)
- **Class Balance:** Checking target distribution
- **Univariate Analysis:** Distributions of Glucose, BMI, and Age
- **Bivariate Analysis:** Boxplots comparing biomarker levels between Diabetic and Non-Diabetic cohorts
- **Multivariate Analysis:** Correlation matrix heatmap
- **Dimensionality Reduction:** 2D PCA projection
"""))

    cells.append(nbf.v4.new_code_cell("""# Visualizing Class Balance and Key Distributions
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# Class Balance
sns.countplot(x='Outcome', data=df_diab, palette=PALETTE, ax=axes[0])
axes[0].set_title('Outcome Class Balance (0: Non-Diabetic, 1: Diabetic)')
axes[0].set_xticklabels(['Non-Diabetic (500)', 'Diabetic (268)'])

# Glucose Distribution
sns.histplot(df_diab_cleaned['Glucose'].dropna(), kde=True, color='#0F766E', ax=axes[1])
axes[1].set_title('Glucose Distribution (Zeros as NaN)')
axes[1].set_xlabel('Glucose (mg/dL)')

# BMI Distribution
sns.histplot(df_diab_cleaned['BMI'].dropna(), kde=True, color='#2563EB', ax=axes[2])
axes[2].set_title('BMI Distribution (Zeros as NaN)')
axes[2].set_xlabel('BMI (kg/m²)')

plt.tight_layout()
plt.show()
"""))

    cells.append(nbf.v4.new_code_cell("""# Bivariate Boxplots: Glucose & Insulin by Outcome
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
sns.boxplot(x='Outcome', y='Glucose', data=df_diab_cleaned, palette=PALETTE, ax=axes[0])
axes[0].set_title('Glucose by Outcome (Higher in Diabetic)')
axes[0].set_xticklabels(['Non-Diabetic', 'Diabetic'])

sns.boxplot(x='Outcome', y='Insulin', data=df_diab_cleaned, palette=PALETTE, ax=axes[1])
axes[1].set_title('Serum Insulin by Outcome')
axes[1].set_xticklabels(['Non-Diabetic', 'Diabetic'])

plt.tight_layout()
plt.show()
"""))

    cells.append(nbf.v4.new_code_cell("""# Multivariate Correlation Heatmap & 2D PCA
fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))

# Correlation Heatmap
corr = df_diab_cleaned.corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='Blues', cbar=True, ax=axes[0], linewidths=0.5)
axes[0].set_title('Diabetes Feature Correlation Heatmap')

# 2D PCA Projection
imputer = SimpleImputer(strategy='median')
scaler = StandardScaler()
X_diab_imputed = scaler.fit_transform(imputer.fit_transform(df_diab_cleaned.drop(columns=['Outcome'])))
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_diab_imputed)

scatter = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=df_diab['Outcome'], cmap='coolwarm', alpha=0.7, edgecolors='k', s=40)
axes[1].set_title(f'Diabetes 2D PCA (Variance Explained: {pca.explained_variance_ratio_.sum():.1%})')
axes[1].set_xlabel('Principal Component 1')
axes[1].set_ylabel('Principal Component 2')
plt.colorbar(scatter, ax=axes[1], label='Outcome (0: Healthy, 1: Diabetic)')

plt.tight_layout()
plt.show()
"""))

    # UNIT I: Heart Disease
    cells.append(nbf.v4.new_markdown_cell("""---
## 2. Heart Disease Dataset: Inspection & Preprocessing (UNIT I)
**Source:** `data/heart.csv` (918 rows)  
**Target:** `HeartDisease` (1 = Disease, 0 = Normal)  
**Data Quality Fixes:**
- `Cholesterol` has 172 zero entries and `RestingBP` has 1 zero entry. These physiologically impossible values are converted to `NaN` and imputed with the median.
- `Oldpeak` (ST depression) contains 13 negative values representing noise/lead reversal artifacts; per domain protocol, these are clipped to 0.0.
- Categorical features: `Sex`, `ChestPainType`, `RestingECG`, `ExerciseAngina`, `ST_Slope` are one-hot encoded.
"""))

    cells.append(nbf.v4.new_code_cell("""# 2.1 Load and Inspect Heart Disease Data
df_heart = pd.read_csv('../data/heart.csv')
print("Heart Dataset Shape:", df_heart.shape)
print("\\n--- Head ---")
display(df_heart.head())

print("\\n--- Info ---")
df_heart.info()

print("\\n--- Zeros and Anomalies Inspection ---")
print("Cholesterol == 0 count:", (df_heart['Cholesterol'] == 0).sum())
print("RestingBP == 0 count:", (df_heart['RestingBP'] == 0).sum())
print("Oldpeak < 0 count:", (df_heart['Oldpeak'] < 0).sum())
print("\\nClass balance:\\n", df_heart['HeartDisease'].value_counts())
"""))

    cells.append(nbf.v4.new_code_cell("""# 2.2 Apply Data Quality Fixes to Heart Data
df_heart_clean = df_heart.copy()
# Convert zeros to NaN
df_heart_clean['Cholesterol'] = df_heart_clean['Cholesterol'].replace(0, np.nan)
df_heart_clean['RestingBP'] = df_heart_clean['RestingBP'].replace(0, np.nan)

# Document clipping of negative Oldpeak values
print("Oldpeak negative values before clipping:\\n", df_heart_clean.loc[df_heart_clean['Oldpeak'] < 0, 'Oldpeak'].tolist())
df_heart_clean['Oldpeak'] = df_heart_clean['Oldpeak'].clip(lower=0.0)
print("Min Oldpeak after clipping:", df_heart_clean['Oldpeak'].min())
"""))

    # UNIT II: Heart Disease EDA
    cells.append(nbf.v4.new_markdown_cell("""### 2.3 Heart Disease Exploratory Data Analysis (UNIT II)
- **Categorical distributions by Heart Disease:** ChestPainType, ST_Slope, Sex
- **Numerical Boxplots:** MaxHR, Age, Cholesterol across classes
- **Multivariate Correlation & 2D PCA**
"""))

    cells.append(nbf.v4.new_code_cell("""# Categorical distributions by Heart Disease
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

sns.countplot(x='ChestPainType', hue='HeartDisease', data=df_heart, palette=PALETTE, ax=axes[0])
axes[0].set_title('Chest Pain Type vs Heart Disease')
axes[0].set_xticklabels(['ATA', 'NAP', 'ASY', 'TA'])

sns.countplot(x='ST_Slope', hue='HeartDisease', data=df_heart, palette=PALETTE, ax=axes[1])
axes[1].set_title('ST Slope vs Heart Disease')

sns.countplot(x='Sex', hue='HeartDisease', data=df_heart, palette=PALETTE, ax=axes[2])
axes[2].set_title('Sex vs Heart Disease')

plt.tight_layout()
plt.show()
"""))

    cells.append(nbf.v4.new_code_cell("""# Numerical features comparison & 2D PCA
fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))

# MaxHR Boxplot
sns.boxplot(x='HeartDisease', y='MaxHR', data=df_heart, palette=PALETTE, ax=axes[0])
axes[0].set_title('MaxHR by Heart Disease Status (Lower in Disease)')
axes[0].set_xticklabels(['Normal (410)', 'Heart Disease (508)'])

# Heart Disease 2D PCA
num_cols_h = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']
cat_cols_h = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']

ct_heart = ColumnTransformer([
    ('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('scl', StandardScaler())]), num_cols_h),
    ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(sparse_output=False))]), cat_cols_h)
])

X_h_prep = ct_heart.fit_transform(df_heart_clean.drop(columns=['HeartDisease']))
pca_h = PCA(n_components=2, random_state=42)
X_h_pca = pca_h.fit_transform(X_h_prep)

scatter = axes[1].scatter(X_h_pca[:, 0], X_h_pca[:, 1], c=df_heart['HeartDisease'], cmap='coolwarm', alpha=0.7, edgecolors='k', s=35)
axes[1].set_title(f'Heart Disease 2D PCA (Variance: {pca_h.explained_variance_ratio_.sum():.1%})')
axes[1].set_xlabel('PC 1')
axes[1].set_ylabel('PC 2')
plt.colorbar(scatter, ax=axes[1], label='Heart Disease')

plt.tight_layout()
plt.show()
"""))

    # UNIT I: Chronic Kidney Disease
    cells.append(nbf.v4.new_markdown_cell("""---
## 3. Chronic Kidney Disease: Cleaning & Preprocessing (UNIT I)
**Source:** `data/kidney_disease.csv` (400 rows)  
**Target:** `classification` ('ckd' = 1, 'notckd' = 0)  
**Data Anomalies Handled:**
- Drop surrogate index column `id`.
- Strip whitespace and stray tab tokens (`'ckd\\t'` -> `'ckd'`).
- Columns `pcv`, `wc`, and `rc` were read as string due to entries containing `\\t?` and `?`. These are cast to float with errors coerced to `NaN`.
- Categoricals with whitespace variations (e.g. `'\\tyes'`, `' yes'`, `'\\tno'`) are normalized to binary tokens.
"""))

    cells.append(nbf.v4.new_code_cell("""# 3.1 Load and Inspect Raw Kidney Data
df_kidney = pd.read_csv('../data/kidney_disease.csv')
print("Kidney Dataset Raw Shape:", df_kidney.shape)
print("Classification Unique Values:", df_kidney['classification'].unique())
print("pcv Unique Values (sample):", df_kidney['pcv'].dropna().unique()[:6])
print("dm Unique Values:", df_kidney['dm'].unique())
"""))

    cells.append(nbf.v4.new_code_cell("""# 3.2 Clean Stray Strings and Convert Data Types
df_kid_clean = df_kidney.copy()
if 'id' in df_kid_clean.columns:
    df_kid_clean.drop(columns=['id'], inplace=True)

# Clean classification target
df_kid_clean['classification'] = df_kid_clean['classification'].astype(str).str.strip().replace({'ckd\\t': 'ckd'})
df_kid_clean = df_kid_clean[df_kid_clean['classification'].isin(['ckd', 'notckd'])].copy()
df_kid_clean['Target'] = df_kid_clean['classification'].map({'ckd': 1, 'notckd': 0})

# Cast numeric columns misread as string
for col in ['pcv', 'wc', 'rc']:
    df_kid_clean[col] = pd.to_numeric(
        df_kid_clean[col].astype(str).str.strip().replace({'?': np.nan, '\\t?': np.nan, 'nan': np.nan}),
        errors='coerce'
    )

# Clean categorical tokens
cat_cols_k = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
for col in cat_cols_k:
    df_kid_clean[col] = df_kid_clean[col].astype(str).str.strip().replace({
        '?': np.nan, '\\t?': np.nan, 'nan': np.nan,
        '\\tno': 'no', '\\tyes': 'yes', ' yes': 'yes'
    })

print("Cleaned Kidney Dataset Shape:", df_kid_clean.shape)
print("Target Distribution:\\n", df_kid_clean['Target'].value_counts())
"""))

    # UNIT II: Kidney Disease EDA
    cells.append(nbf.v4.new_markdown_cell("""### 3.3 Chronic Kidney Disease EDA (UNIT II)
- **Key Biomarkers:** Hemoglobin (`hemo`), Serum Creatinine (`sc`), Blood Urea (`bu`)
- **Correlation Heatmap & 2D PCA**
"""))

    cells.append(nbf.v4.new_code_cell("""# Kidney Biomarkers Analysis
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

sns.boxplot(x='Target', y='hemo', data=df_kid_clean, palette=PALETTE, ax=axes[0])
axes[0].set_title('Hemoglobin by CKD Status (Markedly Lower in CKD)')
axes[0].set_xticklabels(['Not CKD', 'CKD'])

sns.boxplot(x='Target', y='sc', data=df_kid_clean, palette=PALETTE, ax=axes[1])
axes[1].set_title('Serum Creatinine by CKD Status (Elevated in CKD)')
axes[1].set_xticklabels(['Not CKD', 'CKD'])

sns.countplot(x='htn', hue='Target', data=df_kid_clean, palette=PALETTE, ax=axes[2])
axes[2].set_title('Hypertension vs CKD')

plt.tight_layout()
plt.show()
"""))

    cells.append(nbf.v4.new_code_cell("""# Kidney Disease 2D PCA
num_cols_k = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']
ct_k = ColumnTransformer([
    ('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('scl', StandardScaler())]), num_cols_k),
    ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(sparse_output=False))]), cat_cols_k)
])

X_k_prep = ct_k.fit_transform(df_kid_clean.drop(columns=['classification', 'Target']))
pca_k = PCA(n_components=2, random_state=42)
X_k_pca = pca_k.fit_transform(X_k_prep)

plt.figure(figsize=(7, 5))
plt.scatter(X_k_pca[:, 0], X_k_pca[:, 1], c=df_kid_clean['Target'], cmap='coolwarm', alpha=0.75, edgecolors='k', s=40)
plt.title(f'Chronic Kidney Disease 2D PCA (Variance: {pca_k.explained_variance_ratio_.sum():.1%})')
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.colorbar(label='CKD (0: Not CKD, 1: CKD)')
plt.tight_layout()
plt.show()
"""))

    # UNIT I: Breast Cancer
    cells.append(nbf.v4.new_markdown_cell("""---
## 4. Breast Cancer Dataset: Inspection & Preprocessing (UNIT I & II)
**Source:** Scikit-Learn `load_breast_cancer` (569 rows, 30 continuous features)  
**Target:** In scikit-learn, 0 = Malignant, 1 = Benign. To ensure clinical consistency across all models (where 1 = presence of disease/malignancy and 0 = benign/healthy), we set `Target = (raw_target == 0).astype(int)`.
"""))

    cells.append(nbf.v4.new_code_cell("""# 4.1 Load and Inspect Breast Cancer Data
bc = load_breast_cancer(as_frame=True)
df_cancer = bc.frame.copy()
# Map to 1 = Malignant (High Risk Cancer), 0 = Benign (Low Risk)
df_cancer['Risk_Target'] = (df_cancer['target'] == 0).astype(int)

print("Breast Cancer Shape:", df_cancer.shape)
print("Target Breakdown (1: Malignant, 0: Benign):\\n", df_cancer['Risk_Target'].value_counts())
print("\\nMissing Values Total:", df_cancer.isna().sum().sum())
display(df_cancer[['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean concave points', 'Risk_Target']].head())
"""))

    cells.append(nbf.v4.new_code_cell("""# 4.2 Breast Cancer Key Features & 2D PCA Projection
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

sns.boxplot(x='Risk_Target', y='mean radius', data=df_cancer, palette=PALETTE, ax=axes[0])
axes[0].set_title('Mean Radius by Malignancy')
axes[0].set_xticklabels(['Benign (357)', 'Malignant (212)'])

sns.boxplot(x='Risk_Target', y='mean concave points', data=df_cancer, palette=PALETTE, ax=axes[1])
axes[1].set_title('Mean Concave Points by Malignancy')
axes[1].set_xticklabels(['Benign', 'Malignant'])

# 2D PCA
scaler_bc = StandardScaler()
X_bc_scaled = scaler_bc.fit_transform(bc.data)
pca_bc = PCA(n_components=2, random_state=42)
X_bc_pca = pca_bc.fit_transform(X_bc_scaled)

scatter = axes[2].scatter(X_bc_pca[:, 0], X_bc_pca[:, 1], c=df_cancer['Risk_Target'], cmap='coolwarm', alpha=0.75, edgecolors='k', s=35)
axes[2].set_title(f'Breast Cancer 2D PCA ({pca_bc.explained_variance_ratio_.sum():.1%} Variance)')
axes[2].set_xlabel('PC 1')
axes[2].set_ylabel('PC 2')
plt.colorbar(scatter, ax=axes[2], label='1: Malignant, 0: Benign')

plt.tight_layout()
plt.show()
"""))

    # Summary of Unit I-II
    cells.append(nbf.v4.new_markdown_cell("""---
## 5. Summary & Scikit-Learn Pipeline Architecture (UNIT II)
All four datasets have been audited, cleaned, and verified.
Each preprocessing step is encapsulated inside a reproducible `ColumnTransformer` combined with `MedicalDataQualityFixer` inside `src/preprocess.py`.
This guarantees that exact same transformations, imputation values, and encodings are applied at training and inference time without data leakage.
"""))

    nb['cells'] = cells
    return nb

if __name__ == '__main__':
    nb = create_eda_notebook()
    with open('notebooks/01_eda_preprocessing.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print("Created notebooks/01_eda_preprocessing.ipynb successfully!")
