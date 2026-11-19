"""
scripts/build_notebook_02.py
Constructs and executes notebooks/02_models.ipynb
"""

import nbformat as nbf

def create_models_notebook():
    nb = nbf.v4.new_notebook()
    cells = []

    # Title & Metadata
    cells.append(nbf.v4.new_markdown_cell("""# Multi-Disease Risk Prediction System
## Notebook 02: Supervised, Unsupervised, Probabilistic Learning & Generative AI Explanation
**Academic Coursework:** Machine Learning Project  
**Syllabus Coverage:** **UNIT III, UNIT IV & UNIT V**  
- **UNIT III:** Supervised Learning (kNN, Logistic Regression, Decision Tree, Random Forest) & Evaluation Metrics (Accuracy, Precision, Recall, F1, Confusion Matrix, ROC curves, Recall prioritization)
- **UNIT IV:** Unsupervised & Probabilistic Learning (k-Means, Elbow & Silhouette, Hierarchical Clustering Dendrogram, Gaussian Naive Bayes)
- **UNIT V:** Generative AI / LLM Explanation (Plain-language clinical risk synthesis with Gemini API & Offline Fallback)
"""))

    # Imports Cell
    cells.append(nbf.v4.new_code_cell("""# Imports & Configuration
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, silhouette_score
)

# Add parent directory to path to access src package
sys.path.append(os.path.abspath('..'))
from src.preprocess import load_data, split_data, build_full_pipeline, DISEASE_METADATA
from src.train import tune_knn_k, train_and_evaluate_disease
from src.predict import predict_risk, what_if_simulation
from src.explain import explain_prediction

# Visualization styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#E5E7EB'
PRIMARY_COLOR = '#0F766E'
ACCENT_COLOR = '#D97706'
DANGER_COLOR = '#DC2626'
PALETTE = ['#0F766E', '#DC2626']

print("Notebook environment configured successfully!")
"""))

    # UNIT III: Supervised Learning on Diabetes
    cells.append(nbf.v4.new_markdown_cell("""---
## 1. Supervised Learning: Diabetes Risk Modeling (UNIT III)
### 1.1 Hyperparameter Tuning: kNN (Accuracy vs K)
We evaluate $K \\in [1, 25]$ using 5-fold cross-validation on the preprocessed training set.
"""))

    cells.append(nbf.v4.new_code_cell("""# Evaluate kNN across K values
from src.preprocess import build_column_preprocessor

X_diab, y_diab = load_data('diabetes')
X_d_train, X_d_test, y_d_train, y_d_test = split_data(X_diab, y_diab, test_size=0.2, random_state=42)
preproc_diab = build_column_preprocessor('diabetes')

best_k, k_scores = tune_knn_k(X_d_train, y_d_train, preproc_diab)

# Plot Accuracy vs K
plt.figure(figsize=(9, 4.5))
plt.plot(list(k_scores.keys()), list(k_scores.values()), marker='o', color=PRIMARY_COLOR, linewidth=2, markersize=6)
plt.axvline(best_k, color=DANGER_COLOR, linestyle='--', label=f'Optimal K = {best_k} (Acc = {k_scores[best_k]:.3f})')
plt.title('kNN Hyperparameter Tuning: 5-Fold CV Accuracy vs K Value (Diabetes)', fontsize=13)
plt.xlabel('Number of Neighbors (K)', fontsize=11)
plt.ylabel('Mean 5-Fold CV Accuracy', fontsize=11)
plt.xticks(list(k_scores.keys()))
plt.legend(frameon=True)
plt.tight_layout()
plt.show()
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 1.2 Model Comparison & Medical Recall Prioritization
In clinical diagnostics, **Recall (Sensitivity)** is prioritized over simple accuracy.
$$\\text{Recall} = \\frac{\\text{True Positives}}{\\text{True Positives} + \\text{False Negatives}}$$
A **False Negative** means a diseased individual is told they are healthy, delaying critical medical intervention. In contrast, a **False Positive** triggers an inexpensive follow-up test (e.g. repeated fasting plasma glucose test or HbA1c panel). Therefore, we prioritize Recall as the leading decision criterion.
"""))

    cells.append(nbf.v4.new_code_cell("""# Train and evaluate all 5 models on Diabetes
art_diab = train_and_evaluate_disease('diabetes', models_dir='../models')
print("\\n--- Model Comparison Table (Diabetes) ---")
display(art_diab['metrics_df'])
print("\\nSelected Best Model:", art_diab['best_model_name'])
print("Clinical Justification:\\n", art_diab['selection_reason'])
"""))

    cells.append(nbf.v4.new_code_cell("""# ROC Curves and Confusion Matrix for Best Model (Diabetes)
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Multi-model ROC Curves
for name, roc_data in art_diab['roc_curves'].items():
    axes[0].plot(roc_data['fpr'], roc_data['tpr'], label=f"{name} (AUC = {roc_data['auc']:.2f})")
axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
axes[0].set_title('Diabetes: Multi-Model ROC Curves', fontsize=12)
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate (Recall)')
axes[0].legend(loc='lower right', fontsize=9)

# Confusion Matrix for Best Model
best_name = art_diab['best_model_name']
cm = np.array(art_diab['confusion_matrices'][best_name])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1],
            xticklabels=['Pred Healthy', 'Pred Diabetic'],
            yticklabels=['Actual Healthy', 'Actual Diabetic'])
axes[1].set_title(f'Diabetes: Confusion Matrix ({best_name})', fontsize=12)

plt.tight_layout()
plt.show()
"""))

    # UNIT III: Multi-Disease Comparison
    cells.append(nbf.v4.new_markdown_cell("""---
## 2. Supervised Learning Across All Diseases (Heart, Kidney, Cancer)
We train and evaluate the full suite of 5 models on the remaining 3 diseases:
"""))

    cells.append(nbf.v4.new_code_cell("""# Train remaining diseases and display comparison tables
diseases = ['heart', 'kidney', 'cancer']
artifacts = {'diabetes': art_diab}

for d in diseases:
    art = train_and_evaluate_disease(d, models_dir='../models')
    artifacts[d] = art
    print(f"\\n=== {d.upper()} MODEL COMPARISON TABLE ===")
    display(art['metrics_df'])
    print(f"Best Model for {d.upper()}: {art['best_model_name']} (Recall = {art['metrics_df'].loc[art['metrics_df']['Model'] == art['best_model_name'], 'Recall'].values[0]:.1%})")
"""))

    # UNIT IV: Unsupervised & Probabilistic Learning
    cells.append(nbf.v4.new_markdown_cell("""---
## 3. Unsupervised Learning: Risk Clustering & Dendrograms (UNIT IV)
### 3.1 k-Means Clustering: Elbow Method & Silhouette Analysis
We determine optimal cluster counts by inspecting the sum of squared distances (Inertia) and the Silhouette coefficient for $K \\in [2, 6]$.
"""))

    cells.append(nbf.v4.new_code_cell("""# Elbow and Silhouette plots for Diabetes and Heart Disease
fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))

k_eval_diab = art_diab['kmeans_eval']
axes[0].plot(k_eval_diab['k_range'], k_eval_diab['inertia'], marker='o', color=PRIMARY_COLOR, label='Inertia (Elbow)')
axes[0].set_title('Diabetes: k-Means Inertia (Elbow Method)', fontsize=12)
axes[0].set_xlabel('Number of Clusters (K)')
axes[0].set_ylabel('Inertia (Sum of Squared Distances)')

# Silhouette score on second axis
ax2 = axes[0].twinx()
ax2.plot(k_eval_diab['k_range'], k_eval_diab['silhouette'], marker='s', color=ACCENT_COLOR, linestyle='--', label='Silhouette Score')
ax2.set_ylabel('Silhouette Score')
ax2.grid(False)

# Heart Disease Elbow & Silhouette
art_heart = artifacts['heart']
k_eval_heart = art_heart['kmeans_eval']
axes[1].plot(k_eval_heart['k_range'], k_eval_heart['inertia'], marker='o', color=PRIMARY_COLOR, label='Inertia')
axes[1].set_title('Heart Disease: k-Means Inertia & Silhouette', fontsize=12)
axes[1].set_xlabel('Number of Clusters (K)')
axes[1].set_ylabel('Inertia')

ax3 = axes[1].twinx()
ax3.plot(k_eval_heart['k_range'], k_eval_heart['silhouette'], marker='s', color=ACCENT_COLOR, linestyle='--', label='Silhouette')
ax3.set_ylabel('Silhouette Score')
ax3.grid(False)

plt.tight_layout()
plt.show()
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 3.2 Cluster Risk Profiling
Clusters formed by k-Means ($K=3$) are labeled as **Low Risk**, **Moderate Risk**, or **High Risk** according to the empirical proportion of positive disease cases belonging to each cluster.
"""))

    cells.append(nbf.v4.new_code_cell("""# Display Cluster Risk Profiles
for d_name in ['diabetes', 'heart', 'kidney', 'cancer']:
    print(f"\\n=== {d_name.upper()} RISK CLUSTERS ===")
    cmap = artifacts[d_name]['cluster_risk_map']
    for c_id, c_data in cmap.items():
        print(f"Cluster {c_id}: {c_data.get('label', 'Cluster')} -> Size={c_data['size']}, Positives={c_data['positive_count']} ({c_data['positive_rate']:.1%} prevalence)")
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 3.3 Hierarchical Clustering Dendrogram
Using Ward's minimum variance method on preprocessed clinical feature spaces, we visualize sample patient dendrograms.
"""))

    cells.append(nbf.v4.new_code_cell("""# Hierarchical Clustering Dendrogram (Diabetes & Heart)
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

dendrogram(art_diab['linkage_matrix'], ax=axes[0], truncate_mode='lastp', p=25, leaf_rotation=90, leaf_font_size=8)
axes[0].set_title('Diabetes: Hierarchical Clustering Dendrogram (Ward Linkage)', fontsize=12)
axes[0].set_xlabel('Patient Clusters / Indices')
axes[0].set_ylabel('Cluster Distance')

dendrogram(art_heart['linkage_matrix'], ax=axes[1], truncate_mode='lastp', p=25, leaf_rotation=90, leaf_font_size=8)
axes[1].set_title('Heart Disease: Hierarchical Clustering Dendrogram (Ward Linkage)', fontsize=12)
axes[1].set_xlabel('Patient Clusters / Indices')
axes[1].set_ylabel('Cluster Distance')

plt.tight_layout()
plt.show()
"""))

    # UNIT IV: Probabilistic Learning
    cells.append(nbf.v4.new_markdown_cell("""### 3.4 Probabilistic Learning: Gaussian Naive Bayes Deep Dive
Gaussian Naive Bayes applies Bayes' Theorem with the naive assumption of conditional independence between features given the class:
$$P(Y = c \\mid X) = \\frac{P(Y = c) \\prod_{i=1}^n P(X_i \\mid Y = c)}{P(X)}$$
Where $P(X_i \\mid Y = c) \\sim \\mathcal{N}(\\mu_{c,i}, \\sigma_{c,i}^2)$.
"""))

    cells.append(nbf.v4.new_code_cell("""# Gaussian Naive Bayes Class Probabilities for a Sample Patient
sample_patient = {
    'Pregnancies': 3,
    'Glucose': 165.0,
    'BloodPressure': 82.0,
    'SkinThickness': 32.0,
    'Insulin': 160.0,
    'BMI': 35.8,
    'DiabetesPedigreeFunction': 0.62,
    'Age': 48
}

gnb_pipe = art_diab['all_models']['Gaussian Naive Bayes']
sample_df = pd.DataFrame([sample_patient])
gnb_proba = gnb_pipe.predict_proba(sample_df)[0]

print("Gaussian Naive Bayes Probabilities:")
print(f" - P(Healthy | X) = {gnb_proba[0]:.4f} ({gnb_proba[0]*100:.1f}%)")
print(f" - P(Diabetic | X) = {gnb_proba[1]:.4f} ({gnb_proba[1]*100:.1f}%)")
"""))

    # UNIT V: Generative AI Explanation
    cells.append(nbf.v4.new_markdown_cell("""---
## 4. Generative AI / LLM Clinical Explanation (UNIT V)
We synthesize the quantitative outputs (supervised risk percentage, feature contributions, and unsupervised risk cluster) into an empathetic, clear, plain-language patient report.
- Powered by **Google Generative AI (Gemini)** when `GEMINI_API_KEY` is present.
- Equipped with a **Deterministic Clinical Fallback Engine** so the pipeline never breaks if offline.
"""))

    cells.append(nbf.v4.new_code_cell("""# Run Prediction & Generative Explanation
prediction_out = predict_risk('diabetes', sample_patient, models_dir='../models')
explanation_out = explain_prediction('diabetes', sample_patient, prediction_out)

print("Prediction Summary:")
print(f" - Risk Level: {prediction_out['risk_level']} ({prediction_out['risk_percentage']}%)")
print(f" - Risk Cluster: {prediction_out['cluster_label']}")
print(f" - Model Used: {prediction_out['model_name']}")
print(f" - Explanation Source: {explanation_out['source']}")

print("\\n--- Generated Clinical Report ---")
from IPython.display import Markdown
display(Markdown(explanation_out['explanation']))
"""))

    cells.append(nbf.v4.new_code_cell("""# Interactive What-If Sensitivity Test
what_if_res = what_if_simulation('diabetes', sample_patient, {'Glucose': 105.0, 'BMI': 24.0}, models_dir='../models')
print("What-If Simulation Results:")
print(f" - Baseline Risk: {what_if_res['base_risk_pct']}%")
print(f" - Modified Risk: {what_if_res['modified_risk_pct']}%")
print(f" - Absolute Risk Reduction: {what_if_res['delta_pct']}%")
"""))

    nb['cells'] = cells
    return nb

if __name__ == '__main__':
    nb = create_models_notebook()
    with open('notebooks/02_models.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print("Created notebooks/02_models.ipynb successfully!")
