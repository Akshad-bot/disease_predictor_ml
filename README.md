# Multi-Disease Risk Prediction System

A comprehensive Machine Learning project built for academic coursework and clinical decision support. This system predicts risk for four major chronic conditions (**Diabetes, Heart Disease, Chronic Kidney Disease, and Breast Cancer**) using supervised, unsupervised, and probabilistic learning pipelines coupled with Generative AI (LLM) plain-language explanations.

---

## 1. Syllabus & Curriculum Coverage

Each unit of the machine learning curriculum is mapped to dedicated codebase modules and interactive dashboard components:

| Unit | Syllabus Topics | Project Implementation & Design | Codebase / Artifact Reference |
| :--- | :--- | :--- | :--- |
| **UNIT I** | Data Understanding, Cleaning & Domain Imputation | - **Diabetes**: Identified 0s in Glucose, BP, SkinThickness, Insulin, and BMI as missing values and imputed with median.<br>- **Heart Disease**: Addressed 172 zeros in Cholesterol and 1 zero in RestingBP as missing; clipped 13 negative Oldpeak values to 0.<br>- **Kidney Disease**: Stripped stray tabs (`ckd\t`), normalized whitespace, dropped `id`, coerced misread numeric columns (`pcv`, `wc`, `rc`), and handled `?`.<br>- **Breast Cancer**: Inverted target (1: Malignant / High Risk, 0: Benign / Low Risk) for unified medical risk semantics. | [src/preprocess.py](file:///c:/Users/Admin/disease-predictor/src/preprocess.py)<br>[notebooks/01_eda_preprocessing.ipynb](file:///c:/Users/Admin/disease-predictor/notebooks/01_eda_preprocessing.ipynb) |
| **UNIT II** | Preprocessing Pipelines & Exploratory Data Analysis | - Built leak-free scikit-learn `Pipeline` and `ColumnTransformer` (SimpleImputer, StandardScaler, OneHotEncoder).<br>- Univariate distributions, class balance charts, correlation heatmaps, bivariate boxplots, and 2D PCA feature projections. | [src/preprocess.py](file:///c:/Users/Admin/disease-predictor/src/preprocess.py)<br>[notebooks/01_eda_preprocessing.ipynb](file:///c:/Users/Admin/disease-predictor/notebooks/01_eda_preprocessing.ipynb)<br>Streamlit Insights Tab |
| **UNIT III** | Supervised Learning & Clinical Evaluation | - Trained and cross-validated 5 classifiers per disease: **kNN (with K vs Accuracy tuning)**, **Logistic Regression**, **Decision Tree**, **Random Forest**, **Gaussian Naive Bayes**.<br>- 5-Fold Stratified Cross Validation.<br>- Evaluated with Accuracy, Precision, Recall, F1-Score, and ROC-AUC.<br>- **Medical Rationale**: Prioritized **Recall (Sensitivity)** to prevent dangerous False Negatives. | [src/train.py](file:///c:/Users/Admin/disease-predictor/src/train.py)<br>[notebooks/02_models.ipynb](file:///c:/Users/Admin/disease-predictor/notebooks/02_models.ipynb)<br>Streamlit Models Tab |
| **UNIT IV** | Unsupervised & Probabilistic Learning | - **k-Means Clustering**: Evaluated Inertia (Elbow) and Silhouette scores ($K=2..6$); stratified into Low, Moderate, and High risk clusters based on empirical disease prevalence.<br>- **Hierarchical Clustering**: Computed Ward's linkage matrix and plotted dendrograms.<br>- **Probabilistic Learning**: Gaussian Naive Bayes class posterior probabilities. | [src/train.py](file:///c:/Users/Admin/disease-predictor/src/train.py)<br>[src/predict.py](file:///c:/Users/Admin/disease-predictor/src/predict.py)<br>[notebooks/02_models.ipynb](file:///c:/Users/Admin/disease-predictor/notebooks/02_models.ipynb) |
| **UNIT V** | Generative AI / LLM Clinical Synthesis | - Synthesizes quantitative outputs into patient-friendly explanations using **Google Generative AI (Gemini API)**.<br>- Includes a deterministic, clinically grounded rule-based fallback engine so the app never breaks offline or without an API key. | [src/explain.py](file:///c:/Users/Admin/disease-predictor/src/explain.py)<br>Streamlit Predict Tab |

---

## 2. Experimental Results & Benchmark Table

Evaluated on held-out test partitions (80/20 stratified split) across all 4 clinical datasets:

| Disease | Candidate Model | Accuracy | Precision | Recall (Primary) | F1-Score | ROC-AUC | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Diabetes** | k-Nearest Neighbors (K=13) | 71.4% | 60.9% | 51.9% | 56.0% | 0.794 | Candidate |
| | Logistic Regression | 70.8% | 60.0% | 50.0% | 54.5% | 0.813 | Candidate |
| | **Decision Tree** | **76.0%** | **63.9%** | **72.2%** | **67.8%** | **0.762** | **Selected (Best Recall)** |
| | Random Forest | 77.3% | 70.2% | 61.1% | 65.3% | 0.818 | Candidate |
| | Gaussian Naive Bayes | 70.1% | 56.7% | 63.0% | 59.6% | 0.765 | Candidate |
| **Heart Disease** | **k-Nearest Neighbors (K=15)** | **89.1%** | **88.0%** | **93.1%** | **90.5%** | **0.936** | **Selected (Best Recall)** |
| | Logistic Regression | 88.6% | 88.6% | 91.2% | 89.9% | 0.934 | Candidate |
| | Decision Tree | 78.8% | 81.2% | 80.4% | 80.8% | 0.846 | Candidate |
| | Random Forest | 88.6% | 87.9% | 92.2% | 90.0% | 0.932 | Candidate |
| | Gaussian Naive Bayes | 87.0% | 89.0% | 87.3% | 88.1% | 0.907 | Candidate |
| **Chronic Kidney** | k-Nearest Neighbors (K=1) | 96.3% | 100.0% | 94.0% | 96.9% | 0.970 | Candidate |
| | Logistic Regression | 98.8% | 100.0% | 98.0% | 99.0% | 1.000 | Candidate |
| | Decision Tree | 98.8% | 100.0% | 98.0% | 99.0% | 0.990 | Candidate |
| | **Random Forest** | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **1.000** | **Selected (Best Recall)** |
| | Gaussian Naive Bayes | 97.5% | 100.0% | 96.0% | 98.0% | 1.000 | Candidate |
| **Breast Cancer** | k-Nearest Neighbors (K=3) | 93.9% | 97.3% | 85.7% | 91.1% | 0.982 | Candidate |
| | Logistic Regression | 96.5% | 97.5% | 92.9% | 95.1% | 0.996 | Candidate |
| | Decision Tree | 92.1% | 94.6% | 83.3% | 88.6% | 0.945 | Candidate |
| | **Random Forest** | **97.4%** | **100.0%** | **92.9%** | **96.3%** | **0.993** | **Selected (Best Recall & F1)** |
| | Gaussian Naive Bayes | 92.1% | 92.3% | 85.7% | 88.9% | 0.989 | Candidate |

### Why Recall is Prioritized in Medical Diagnostics
In general machine learning tasks, accuracy or F1-score is typically maximized. However, clinical decision systems prioritize **Recall (Sensitivity)**:
$$\text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}$$
A **False Negative** represents a patient with active pathology who is incorrectly categorized as healthy. This can cause severe clinical consequences, delayed therapeutics, or disease progression. Conversely, a **False Positive** triggers non-invasive confirmatory testing (e.g. repeated fasting glucose, echocardiography, or renal ultrasound), which carries minimal clinical risk.

---

## 3. Project Directory Architecture

```
disease-predictor/
├── .streamlit/
│   └── config.toml               # Clean healthcare dashboard theme tokens
├── data/
│   ├── diabetes.csv              # Pima Indians diabetes dataset
│   ├── heart.csv                 # Cardiovascular dataset (918 samples)
│   └── kidney_disease.csv        # Chronic kidney disease dataset (400 samples)
├── models/                       # Serialized pipelines & artifacts (.joblib)
│   ├── cancer_model.joblib
│   ├── diabetes_model.joblib
│   ├── heart_model.joblib
│   └── kidney_model.joblib
├── notebooks/
│   ├── 01_eda_preprocessing.ipynb # UNIT I & II: Preprocessing & Comprehensive EDA
│   └── 02_models.ipynb            # UNIT III, IV & V: ML Models, Clustering & LLM
├── scripts/
│   ├── build_notebook_01.py      # Automated generator for Notebook 01
│   └── build_notebook_02.py      # Automated generator for Notebook 02
├── src/
│   ├── __init__.py
│   ├── preprocess.py             # Data loading, domain fixes, ColumnTransformer
│   ├── train.py                  # 5-model training, 5-fold CV, k-Means, hierarchical
│   ├── predict.py                # Inference, what-if simulator, top feature impact
│   └── explain.py                # Gemini LLM plain-language explanation & fallback
├── app.py                        # Streamlit clinical dashboard application
├── requirements.txt              # Project environment dependencies
└── README.md                     # Comprehensive project documentation
```

---

## 4. Setup & Execution Instructions

### Prerequisites
The environment uses Python 3.10+ (tested on Python 3.13) with the packages specified in `requirements.txt`.

### Step 1: Clone and Enter Directory
```bash
cd disease-predictor
```

### Step 2: (Optional) Set Generative AI API Key
The system uses Google Gemini for real-time generative clinical explanations. If you have a key:
```powershell
$env:GEMINI_API_KEY="your-gemini-api-key"
```
*Note: If no key is set, the system automatically falls back to its built-in Clinical Rule Engine. The application never crashes.*

### Step 3: Run the Streamlit Application
```bash
streamlit run app.py
```
Open your browser to `http://localhost:8501`.

### Step 4: Run the Jupyter Notebooks
To explore the step-by-step code and plots:
```bash
jupyter notebook notebooks/01_eda_preprocessing.ipynb
jupyter notebook notebooks/02_models.ipynb
```

---

## 5. Key Dashboard Features

1. **Disease Stratification Selector:** Seamlessly switch between Diabetes, Heart Disease, Chronic Kidney Disease, and Breast Cancer.
2. **Clinical Input Forms:** Grouped logically into hemodynamic, urinalysis, biochemistry, and demographic panels with full clinical names.
3. **Risk Stratification Panel:** Displays calibrated risk percentage, clinical risk level (`Low Risk`, `Moderate Risk`, `High Risk`), and unsupervised k-Means cluster assignment.
4. **Local Feature Attribution:** Horizontal bar chart highlighting the top 5 contributing factors for the specific patient.
5. **Interactive What-If Sensitivity Simulator:** Live sliders for key modifiable risk factors (e.g. Glucose, BMI, Cholesterol) demonstrating instantaneous risk reductions.
6. **Population Comparison View:** Plots patient biomarker levels relative to healthy and diseased historical cohorts.
7. **Generative Plain-Language Clinical Synthesis:** Produces structured, patient-friendly markdown summaries with actionable next steps.
8. **Clinical Summary Report Download:** One-click download of an official `.txt` risk summary.

---

## 6. Disclaimer
*This system is developed as an educational Machine Learning coursework project. It is intended for academic and decision-support simulation only and does not constitute a formal medical diagnosis. Users should consult licensed healthcare providers for clinical care.*
