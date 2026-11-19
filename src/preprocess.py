"""
src/preprocess.py
=================
UNIT I-II: Data Understanding, Preprocessing, and Pipeline Construction.
Handles data loading, domain-specific quality fixes, feature encoding,
imputation, scaling, and train-test splitting for 4 diseases:
1. Diabetes
2. Heart Disease
3. Breast Cancer
4. Chronic Kidney Disease
"""

import os
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_breast_cancer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def _resolve_path(rel_path: str) -> str:
    """Resolves relative filepaths robustly whether running from root or notebooks."""
    if os.path.isabs(rel_path) and os.path.exists(rel_path):
        return rel_path
    if os.path.exists(rel_path):
        return rel_path
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidate = os.path.join(root_dir, rel_path)
    if os.path.exists(candidate):
        return candidate
    candidate2 = os.path.join("..", rel_path)
    if os.path.exists(candidate2):
        return candidate2
    return rel_path


# -------------------------------------------------------------------------
# Metadata & Definitions for UI and Validation
# -------------------------------------------------------------------------
DISEASE_METADATA = {
    "diabetes": {
        "title": "Diabetes Risk",
        "target_col": "Outcome",
        "target_names": ["Non-Diabetic (Low Risk)", "Diabetic (High Risk)"],
        "numeric_cols": [
            "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
        ],
        "categorical_cols": [],
        "modifiable_cols": ["Glucose", "BMI", "BloodPressure"],
        "feature_labels": {
            "Pregnancies": "Pregnancies (count)",
            "Glucose": "Fasting Plasma Glucose (mg/dL)",
            "BloodPressure": "Diastolic Blood Pressure (mm Hg)",
            "SkinThickness": "Triceps Skinfold Thickness (mm)",
            "Insulin": "2-Hour Serum Insulin (μU/mL)",
            "BMI": "Body Mass Index (kg/m²)",
            "DiabetesPedigreeFunction": "Diabetes Pedigree Genetic Score",
            "Age": "Age (years)",
        },
        "defaults": {
            "Pregnancies": 2,
            "Glucose": 117.0,
            "BloodPressure": 72.0,
            "SkinThickness": 23.0,
            "Insulin": 30.5,
            "BMI": 32.0,
            "DiabetesPedigreeFunction": 0.372,
            "Age": 29,
        },
        "ranges": {
            "Pregnancies": (0, 17, 1),
            "Glucose": (50.0, 250.0, 1.0),
            "BloodPressure": (40.0, 140.0, 1.0),
            "SkinThickness": (5.0, 100.0, 1.0),
            "Insulin": (10.0, 850.0, 1.0),
            "BMI": (15.0, 65.0, 0.1),
            "DiabetesPedigreeFunction": (0.05, 2.50, 0.01),
            "Age": (21, 90, 1),
        },
    },
    "heart": {
        "title": "Heart Disease Risk",
        "target_col": "HeartDisease",
        "target_names": ["Normal / Healthy", "Heart Disease Present"],
        "numeric_cols": [
            "Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"
        ],
        "categorical_cols": [
            "Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"
        ],
        "modifiable_cols": ["Cholesterol", "RestingBP", "MaxHR"],
        "feature_labels": {
            "Age": "Patient Age (years)",
            "Sex": "Biological Sex",
            "ChestPainType": "Chest Pain Type",
            "RestingBP": "Resting Blood Pressure (mm Hg)",
            "Cholesterol": "Serum Cholesterol (mg/dL)",
            "FastingBS": "Fasting Blood Sugar (> 120 mg/dL)",
            "RestingECG": "Resting ECG Results",
            "MaxHR": "Maximum Heart Rate Achieved (bpm)",
            "ExerciseAngina": "Exercise-Induced Angina",
            "Oldpeak": "ST Depression (Oldpeak)",
            "ST_Slope": "Peak Exercise ST Slope",
        },
        "category_options": {
            "Sex": {"M": "Male", "F": "Female"},
            "ChestPainType": {
                "ASY": "Asymptomatic (ASY)",
                "NAP": "Non-Anginal Pain (NAP)",
                "ATA": "Atypical Angina (ATA)",
                "TA": "Typical Angina (TA)",
            },
            "RestingECG": {
                "Normal": "Normal",
                "ST": "ST-T Wave Abnormality",
                "LVH": "Left Ventricular Hypertrophy (LVH)",
            },
            "ExerciseAngina": {"N": "No", "Y": "Yes"},
            "ST_Slope": {
                "Flat": "Flat Slope",
                "Up": "Upsloping",
                "Down": "Downsloping",
            },
            "FastingBS": {0: "Normal (<= 120 mg/dL)", 1: "Elevated (> 120 mg/dL)"},
        },
        "defaults": {
            "Age": 54,
            "Sex": "M",
            "ChestPainType": "ASY",
            "RestingBP": 130.0,
            "Cholesterol": 223.0,
            "FastingBS": 0,
            "RestingECG": "Normal",
            "MaxHR": 138.0,
            "ExerciseAngina": "N",
            "Oldpeak": 0.6,
            "ST_Slope": "Flat",
        },
        "ranges": {
            "Age": (28, 80, 1),
            "RestingBP": (80.0, 200.0, 1.0),
            "Cholesterol": (100.0, 600.0, 1.0),
            "FastingBS": (0, 1, 1),
            "MaxHR": (60.0, 210.0, 1.0),
            "Oldpeak": (0.0, 6.5, 0.1),
        },
    },
    "kidney": {
        "title": "Chronic Kidney Disease Risk",
        "target_col": "classification",
        "target_names": ["Not CKD (Healthy)", "Chronic Kidney Disease (CKD)"],
        "numeric_cols": [
            "age", "bp", "sg", "al", "su", "bgr", "bu", "sc", "sod", "pot",
            "hemo", "pcv", "wc", "rc"
        ],
        "categorical_cols": [
            "rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane"
        ],
        "modifiable_cols": ["bp", "bgr", "bu", "sc"],
        "feature_labels": {
            "age": "Age (years)",
            "bp": "Blood Pressure (mm Hg)",
            "sg": "Specific Gravity",
            "al": "Albumin (0 to 5)",
            "su": "Sugar (0 to 5)",
            "rbc": "Red Blood Cells in Urine",
            "pc": "Pus Cells in Urine",
            "pcc": "Pus Cell Clumps",
            "ba": "Bacteria Presence",
            "bgr": "Blood Glucose Random (mg/dL)",
            "bu": "Blood Urea (mg/dL)",
            "sc": "Serum Creatinine (mg/dL)",
            "sod": "Sodium (mEq/L)",
            "pot": "Potassium (mEq/L)",
            "hemo": "Hemoglobin (g/dL)",
            "pcv": "Packed Cell Volume (%)",
            "wc": "White Blood Cell Count (/μL)",
            "rc": "Red Blood Cell Count (millions/μL)",
            "htn": "Hypertension",
            "dm": "Diabetes Mellitus",
            "cad": "Coronary Artery Disease",
            "appet": "Appetite",
            "pe": "Pedal Edema",
            "ane": "Anemia",
        },
        "category_options": {
            "rbc": {"normal": "Normal", "abnormal": "Abnormal"},
            "pc": {"normal": "Normal", "abnormal": "Abnormal"},
            "pcc": {"notpresent": "Not Present", "present": "Present"},
            "ba": {"notpresent": "Not Present", "present": "Present"},
            "htn": {"no": "No", "yes": "Yes"},
            "dm": {"no": "No", "yes": "Yes"},
            "cad": {"no": "No", "yes": "Yes"},
            "appet": {"good": "Good", "poor": "Poor"},
            "pe": {"no": "No", "yes": "Yes"},
            "ane": {"no": "No", "yes": "Yes"},
        },
        "defaults": {
            "age": 55.0,
            "bp": 80.0,
            "sg": 1.020,
            "al": 0.0,
            "su": 0.0,
            "rbc": "normal",
            "pc": "normal",
            "pcc": "notpresent",
            "ba": "notpresent",
            "bgr": 121.0,
            "bu": 42.0,
            "sc": 1.2,
            "sod": 138.0,
            "pot": 4.4,
            "hemo": 12.5,
            "pcv": 38.0,
            "wc": 8000.0,
            "rc": 4.8,
            "htn": "no",
            "dm": "no",
            "cad": "no",
            "appet": "good",
            "pe": "no",
            "ane": "no",
        },
        "ranges": {
            "age": (1.0, 95.0, 1.0),
            "bp": (50.0, 180.0, 5.0),
            "sg": (1.005, 1.025, 0.005),
            "al": (0.0, 5.0, 1.0),
            "su": (0.0, 5.0, 1.0),
            "bgr": (50.0, 500.0, 1.0),
            "bu": (10.0, 390.0, 1.0),
            "sc": (0.4, 20.0, 0.1),
            "sod": (100.0, 165.0, 1.0),
            "pot": (2.0, 9.0, 0.1),
            "hemo": (3.0, 18.0, 0.1),
            "pcv": (10.0, 60.0, 1.0),
            "wc": (2200.0, 26000.0, 100.0),
            "rc": (2.0, 8.0, 0.1),
        },
    },
    "cancer": {
        "title": "Breast Cancer Risk",
        "target_col": "target",
        "target_names": ["Benign (Low Risk)", "Malignant (High Risk)"],
        "numeric_cols": [
            "mean radius", "mean texture", "mean perimeter", "mean area",
            "mean smoothness", "mean compactness", "mean concavity",
            "mean concave points", "mean symmetry", "mean fractal dimension",
            "radius error", "texture error", "perimeter error", "area error",
            "smoothness error", "compactness error", "concavity error",
            "concave points error", "symmetry error", "fractal dimension error",
            "worst radius", "worst texture", "worst perimeter", "worst area",
            "worst smoothness", "worst compactness", "worst concavity",
            "worst concave points", "worst symmetry", "worst fractal dimension"
        ],
        "categorical_cols": [],
        "modifiable_cols": ["mean radius", "mean texture", "mean concave points"],
        "feature_labels": {
            "mean radius": "Mean Radius (cell nuclei size)",
            "mean texture": "Mean Texture (gray-scale std dev)",
            "mean perimeter": "Mean Perimeter",
            "mean area": "Mean Area",
            "mean smoothness": "Mean Smoothness (local variation in radius)",
            "mean compactness": "Mean Compactness (perimeter² / area - 1.0)",
            "mean concavity": "Mean Concavity (severity of concave portions)",
            "mean concave points": "Mean Concave Points (number of concave portions)",
            "mean symmetry": "Mean Symmetry",
            "mean fractal dimension": "Mean Fractal Dimension",
            "worst radius": "Worst Radius",
            "worst texture": "Worst Texture",
            "worst perimeter": "Worst Perimeter",
            "worst area": "Worst Area",
            "worst smoothness": "Worst Smoothness",
            "worst compactness": "Worst Compactness",
            "worst concavity": "Worst Concavity",
            "worst concave points": "Worst Concave Points",
            "worst symmetry": "Worst Symmetry",
            "worst fractal dimension": "Worst Fractal Dimension",
        },
        "defaults": {},
        "ranges": {},
    },
}

# Pre-populate Breast Cancer medians and ranges at module load time
def _init_cancer_metadata():
    bc = load_breast_cancer(as_frame=True)
    X = bc.data
    medians = X.median()
    mins = X.min()
    maxs = X.max()
    for col in X.columns:
        step = 0.1 if maxs[col] > 10 else 0.01
        DISEASE_METADATA["cancer"]["defaults"][col] = round(float(medians[col]), 4)
        DISEASE_METADATA["cancer"]["ranges"][col] = (
            round(float(mins[col]), 4),
            round(float(maxs[col]), 4),
            step
        )

_init_cancer_metadata()



# -------------------------------------------------------------------------
# Custom Transformer for Preprocessing Data Quality Fixes
# -------------------------------------------------------------------------
class MedicalDataQualityFixer(BaseEstimator, TransformerMixin):
    """
    Applies domain-specific medical data cleaning:
    - Diabetes: zeros in Glucose, BloodPressure, SkinThickness, Insulin, BMI -> NaN
    - Heart: zeros in Cholesterol, RestingBP -> NaN; clip Oldpeak to >= 0
    - Kidney: string whitespace cleaning, converting '?' and stray tokens to NaN
    """
    def __init__(self, disease: str = "diabetes"):
        self.disease = disease

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_df = X.copy()
        if not isinstance(X_df, pd.DataFrame):
            # If passed as numpy array, convert to DataFrame if column names known
            return X_df

        if self.disease == "diabetes":
            zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
            for col in zero_cols:
                if col in X_df.columns:
                    X_df[col] = X_df[col].replace(0, np.nan)

        elif self.disease == "heart":
            if "Cholesterol" in X_df.columns:
                X_df["Cholesterol"] = X_df["Cholesterol"].replace(0, np.nan)
            if "RestingBP" in X_df.columns:
                X_df["RestingBP"] = X_df["RestingBP"].replace(0, np.nan)
            if "Oldpeak" in X_df.columns:
                X_df["Oldpeak"] = pd.to_numeric(X_df["Oldpeak"], errors="coerce").clip(lower=0.0)

        elif self.disease == "kidney":
            # Strip string columns and replace placeholder symbols with NaN
            for col in X_df.columns:
                if X_df[col].dtype == object or X_df[col].dtype.name == "str":
                    X_df[col] = X_df[col].astype(str).str.strip().replace({
                        "?": np.nan,
                        "\t?": np.nan,
                        "nan": np.nan,
                        "\tno": "no",
                        "\tyes": "yes",
                        " yes": "yes",
                    })

        return X_df


# -------------------------------------------------------------------------
# Data Loaders
# -------------------------------------------------------------------------
def load_diabetes_data(filepath: str = "data/diabetes.csv") -> tuple[pd.DataFrame, pd.Series]:
    """
    UNIT I: Load Diabetes dataset.
    Target: 'Outcome' (1: Diabetic, 0: Non-Diabetic)
    Treat zeros in Glucose, BloodPressure, SkinThickness, Insulin, BMI as NaN.
    """
    filepath = _resolve_path(filepath)
    df = pd.read_csv(filepath)
    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]
    return X, y


def load_heart_data(filepath: str = "data/heart.csv") -> tuple[pd.DataFrame, pd.Series]:
    """
    UNIT I: Load Heart Disease dataset (918 rows).
    Target: 'HeartDisease' (1: Disease, 0: Normal)
    Quality fixes: Cholesterol (172 zeros) and RestingBP (1 zero) -> NaN.
    Oldpeak (13 negative values) clipped to 0.
    """
    filepath = _resolve_path(filepath)
    df = pd.read_csv(filepath)
    X = df.drop(columns=["HeartDisease"])
    y = df["HeartDisease"]
    return X, y


def load_kidney_data(filepath: str = "data/kidney_disease.csv") -> tuple[pd.DataFrame, pd.Series]:
    """
    UNIT I: Load Chronic Kidney Disease dataset.
    Target: 'classification' ('ckd' -> 1, 'notckd' -> 0)
    Drops 'id', strips stray tabs/whitespace, converts pcv, wc, rc to numeric.
    """
    filepath = _resolve_path(filepath)
    df = pd.read_csv(filepath)
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    # Clean target
    df["classification"] = df["classification"].astype(str).str.strip()
    df["classification"] = df["classification"].replace({"ckd\t": "ckd"})
    df = df[df["classification"].isin(["ckd", "notckd"])].copy()
    y = df["classification"].map({"ckd": 1, "notckd": 0}).astype(int)
    X = df.drop(columns=["classification"])

    # Clean numeric columns read as string
    for col in ["pcv", "wc", "rc"]:
        if col in X.columns:
            X[col] = pd.to_numeric(
                X[col].astype(str).str.strip().replace({"?": np.nan, "\t?": np.nan, "nan": np.nan}),
                errors="coerce"
            )

    # Clean categoricals
    cat_cols = ["rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane"]
    for col in cat_cols:
        if col in X.columns:
            X[col] = X[col].astype(str).str.strip().replace({
                "?": np.nan,
                "\t?": np.nan,
                "nan": np.nan,
                "\tno": "no",
                "\tyes": "yes",
                " yes": "yes",
            })

    return X, y


def load_breast_cancer_data() -> tuple[pd.DataFrame, pd.Series]:
    """
    UNIT I: Load Breast Cancer dataset from scikit-learn.
    Target: in sklearn, 0 = malignant, 1 = benign.
    To align with standard medical risk models (1 = Malignant/Disease, 0 = Benign/Healthy),
    we map target = (raw_target == 0).astype(int).
    """
    bc = load_breast_cancer(as_frame=True)
    X = bc.data.copy()
    # 1: Malignant (Cancer detected / High Risk), 0: Benign (Healthy / Low Risk)
    y = (bc.target == 0).astype(int)

    # Populate defaults & ranges in metadata
    medians = X.median()
    mins = X.min()
    maxs = X.max()
    for col in X.columns:
        step = 0.1 if maxs[col] > 10 else 0.01
        DISEASE_METADATA["cancer"]["defaults"][col] = round(float(medians[col]), 4)
        DISEASE_METADATA["cancer"]["ranges"][col] = (
            round(float(mins[col]), 4),
            round(float(maxs[col]), 4),
            step
        )

    return X, y


def load_data(disease: str) -> tuple[pd.DataFrame, pd.Series]:
    """Dispatcher to load any of the 4 disease datasets."""
    if disease == "diabetes":
        return load_diabetes_data()
    elif disease == "heart":
        return load_heart_data()
    elif disease == "kidney":
        return load_kidney_data()
    elif disease == "cancer":
        return load_breast_cancer_data()
    else:
        raise ValueError(f"Unknown disease: {disease}. Choose from 'diabetes', 'heart', 'kidney', 'cancer'.")


# -------------------------------------------------------------------------
# Pipeline & Preprocessor Construction (UNIT II)
# -------------------------------------------------------------------------
def build_column_preprocessor(disease: str) -> ColumnTransformer:
    """
    Builds a scikit-learn ColumnTransformer for the specified disease.
    Numeric features: SimpleImputer(median) + StandardScaler()
    Categorical features: SimpleImputer(most_frequent) + OneHotEncoder(handle_unknown='ignore')
    """
    meta = DISEASE_METADATA[disease]
    num_cols = meta["numeric_cols"]
    cat_cols = meta["categorical_cols"]

    transformers = []
    if num_cols:
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num", num_pipeline, num_cols))

    if cat_cols:
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("cat", cat_pipeline, cat_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return preprocessor


def build_full_pipeline(disease: str, classifier) -> Pipeline:
    """
    Builds the end-to-end scikit-learn Pipeline incorporating:
    1. MedicalDataQualityFixer (handles zero-to-NaN conversions, Oldpeak clipping)
    2. ColumnTransformer (imputation + scaling + one-hot encoding)
    3. Classifier (e.g. LogisticRegression, RandomForest, etc.)
    """
    cleaner = MedicalDataQualityFixer(disease=disease)
    preprocessor = build_column_preprocessor(disease=disease)

    pipeline = Pipeline([
        ("cleaner", cleaner),
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])
    return pipeline


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    """
    Stratified 80/20 train/test split.
    """
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
