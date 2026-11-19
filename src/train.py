"""
src/train.py
============
UNIT III: Supervised Learning (kNN, Logistic Regression, Decision Tree,
Random Forest) & Evaluation Metrics.
UNIT IV: Unsupervised & Probabilistic Learning (k-Means, Hierarchical
Clustering, Gaussian Naive Bayes).

Trains and evaluates 5 models per disease, runs clustering, performs
5-fold cross validation, prioritizes Recall for medical screening,
and serializes artifacts with joblib.
"""

import os
import joblib
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    silhouette_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from src.preprocess import (
    DISEASE_METADATA,
    MedicalDataQualityFixer,
    build_column_preprocessor,
    build_full_pipeline,
    load_data,
    split_data,
)


def tune_knn_k(X_train, y_train, preprocessor, k_range=range(1, 26)):
    """
    Find optimal K for kNN using 5-fold cross-validation.
    Returns best_k and a dictionary of {k: cv_mean_accuracy}.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    k_scores = {}
    best_k = 5
    best_score = -1.0

    for k in k_range:
        pipe = Pipeline([
            ("cleaner", MedicalDataQualityFixer()),
            ("preprocessor", preprocessor),
            ("knn", KNeighborsClassifier(n_neighbors=k)),
        ])
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy")
        mean_score = float(np.mean(scores))
        k_scores[k] = mean_score
        if mean_score > best_score:
            best_score = mean_score
            best_k = k

    return best_k, k_scores


from sklearn.pipeline import Pipeline


def train_and_evaluate_disease(disease: str, models_dir: str = "models") -> dict:
    """
    Complete training, evaluation, and clustering workflow for a disease.
    """
    print(f"\n=======================================================")
    print(f"Training models for disease: {disease.upper()}")
    print(f"=======================================================")

    # 1. Load and split data
    X, y = load_data(disease)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)

    meta = DISEASE_METADATA[disease]
    preprocessor = build_column_preprocessor(disease)

    # 2. kNN Hyperparameter Tuning (K vs Accuracy)
    print("Evaluating kNN across K values (1 to 25)...")
    best_k, knn_k_scores = tune_knn_k(X_train, y_train, preprocessor)
    print(f"Optimal K for kNN: {best_k} (CV Accuracy: {knn_k_scores[best_k]:.4f})")

    # 3. Define the 5 candidate models
    classifiers = {
        f"k-Nearest Neighbors (K={best_k})": KNeighborsClassifier(n_neighbors=best_k),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gaussian Naive Bayes": GaussianNB(),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    trained_pipelines = {}
    cv_results = {}
    metrics_records = []
    roc_curves = {}
    confusion_matrices = {}

    for name, clf in classifiers.items():
        pipe = build_full_pipeline(disease, clf)

        # 5-fold Cross-Validation on training set
        cv_recall = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="recall")
        cv_f1 = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1")
        cv_acc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy")
        cv_results[name] = {
            "cv_recall_mean": float(np.mean(cv_recall)),
            "cv_recall_std": float(np.std(cv_recall)),
            "cv_f1_mean": float(np.mean(cv_f1)),
            "cv_acc_mean": float(np.mean(cv_acc)),
        }

        # Train on entire training set
        pipe.fit(X_train, y_train)
        trained_pipelines[name] = pipe

        # Evaluate on held-out test set
        y_pred = pipe.predict(X_test)
        if hasattr(pipe.named_steps["classifier"], "predict_proba"):
            y_proba = pipe.predict_proba(X_test)[:, 1]
        else:
            y_proba = y_pred

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        auc = float(roc_auc_score(y_test, y_proba))

        metrics_records.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "ROC-AUC": auc,
            "CV Recall": float(np.mean(cv_recall)),
        })

        # ROC Curve points
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        roc_curves[name] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "auc": auc,
        }

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        confusion_matrices[name] = cm.tolist()

    metrics_df = pd.DataFrame(metrics_records)
    print("\nModel Evaluation Comparison Table (Test Set):")
    print(metrics_df.to_string(index=False))

    # 4. Model Selection: Prioritize RECALL for medical diagnostic safety
    # A False Negative (missing a patient with the disease) has catastrophic clinical consequences.
    # We sort primarily by Recall, breaking ties by F1-Score and ROC-AUC.
    sorted_metrics = metrics_df.sort_values(by=["Recall", "F1-Score", "ROC-AUC"], ascending=False)
    best_model_name = sorted_metrics.iloc[0]["Model"]
    best_pipeline = trained_pipelines[best_model_name]
    best_recall = sorted_metrics.iloc[0]["Recall"]

    selection_reason = (
        f"Selected {best_model_name} as the primary clinical model because it achieved the "
        f"highest Recall ({best_recall:.1%}) on held-out data. In medical screening, prioritizing Recall "
        f"minimizes dangerous False Negatives (missed diagnoses), while False Positives can be safely "
        f"adjudicated via follow-up diagnostic tests."
    )
    print(f"\nBest Model: {best_model_name}")
    print(f"Reason: {selection_reason}")

    # 5. Extract Feature Names and Feature Importance
    preprocessor_fitted = best_pipeline.named_steps["preprocessor"]
    try:
        feature_names = preprocessor_fitted.get_feature_names_out().tolist()
    except Exception:
        feature_names = meta["numeric_cols"] + meta["categorical_cols"]

    # Calculate global feature importances if available
    classifier_step = best_pipeline.named_steps["classifier"]
    feature_importances = {}
    if hasattr(classifier_step, "feature_importances_"):
        imps = classifier_step.feature_importances_
        feature_importances = dict(zip(feature_names, [float(x) for x in imps]))
    elif hasattr(classifier_step, "coef_"):
        coefs = np.abs(classifier_step.coef_[0])
        feature_importances = dict(zip(feature_names, [float(x) for x in coefs]))
    else:
        # Fall back to Random Forest importances
        rf_step = trained_pipelines["Random Forest"].named_steps["classifier"]
        feature_importances = dict(zip(feature_names, [float(x) for x in rf_step.feature_importances_]))

    # 6. UNIT IV: Unsupervised Learning - k-Means and Hierarchical Clustering
    print("\nFitting Unsupervised Clustering (k-Means & Hierarchical)...")
    cleaner = best_pipeline.named_steps["cleaner"]
    X_cleaned = cleaner.transform(X)
    X_prep = preprocessor_fitted.fit_transform(X_cleaned)

    # Evaluate k-Means for K=2..6 (Elbow & Silhouette)
    kmeans_eval = {"k_range": list(range(2, 7)), "inertia": [], "silhouette": []}
    for k in range(2, 7):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_prep)
        kmeans_eval["inertia"].append(float(km.inertia_))
        sil = silhouette_score(X_prep, km.labels_)
        kmeans_eval["silhouette"].append(float(sil))

    # Fit 3 risk clusters (Low, Moderate, High)
    optimal_k = 3
    kmeans_3 = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = kmeans_3.fit_predict(X_prep)

    # Risk profiling of clusters by share of positive cases
    cluster_risk_map = {}
    for c in range(optimal_k):
        mask = cluster_labels == c
        cluster_size = int(np.sum(mask))
        positive_count = int(np.sum(y[mask] == 1))
        pos_rate = float(positive_count / cluster_size) if cluster_size > 0 else 0.0
        cluster_risk_map[c] = {
            "cluster_id": c,
            "size": cluster_size,
            "positive_count": positive_count,
            "positive_rate": pos_rate,
        }

    # Assign risk tier: sort by positive_rate
    sorted_clusters = sorted(cluster_risk_map.items(), key=lambda item: item[1]["positive_rate"])
    tier_labels = ["Low Risk Cluster", "Moderate Risk Cluster", "High Risk Cluster"]
    for idx, (c_id, info) in enumerate(sorted_clusters):
        cluster_risk_map[c_id]["label"] = tier_labels[idx]
        print(f"Cluster {c_id}: {tier_labels[idx]} ({info['positive_rate']:.1%} disease prevalence, n={info['size']})")

    # Hierarchical Clustering Linkage Matrix (Ward's method, sample up to 150 points for clear dendrogram)
    sample_size = min(150, X_prep.shape[0])
    np.random.seed(42)
    sample_indices = np.random.choice(X_prep.shape[0], sample_size, replace=False)
    X_sample_hierarchical = X_prep[sample_indices]
    linkage_mat = linkage(X_sample_hierarchical, method="ward")

    # 7. Cohort distributions for "Compare to dataset" view
    # Group by healthy (0) and disease (1) for key features
    cohort_summary = {}
    for col in meta["numeric_cols"]:
        if col in X.columns:
            cohort_summary[col] = {
                "healthy_mean": float(X.loc[y == 0, col].mean(skipna=True)),
                "healthy_std": float(X.loc[y == 0, col].std(skipna=True)),
                "healthy_median": float(X.loc[y == 0, col].median(skipna=True)),
                "healthy_p25": float(X.loc[y == 0, col].quantile(0.25)),
                "healthy_p75": float(X.loc[y == 0, col].quantile(0.75)),
                "disease_mean": float(X.loc[y == 1, col].mean(skipna=True)),
                "disease_std": float(X.loc[y == 1, col].std(skipna=True)),
                "disease_median": float(X.loc[y == 1, col].median(skipna=True)),
                "disease_p25": float(X.loc[y == 1, col].quantile(0.25)),
                "disease_p75": float(X.loc[y == 1, col].quantile(0.75)),
            }

    # 8. Package artifact
    artifact = {
        "disease": disease,
        "best_model_name": best_model_name,
        "best_pipeline": best_pipeline,
        "selection_reason": selection_reason,
        "all_models": trained_pipelines,
        "cv_results": cv_results,
        "metrics_df": metrics_df,
        "roc_curves": roc_curves,
        "confusion_matrices": confusion_matrices,
        "knn_k_scores": knn_k_scores,
        "best_k": best_k,
        "kmeans_model": kmeans_3,
        "kmeans_eval": kmeans_eval,
        "cluster_risk_map": cluster_risk_map,
        "linkage_matrix": linkage_mat,
        "feature_names": feature_names,
        "feature_importances": feature_importances,
        "cohort_summary": cohort_summary,
        "metadata": meta,
        "X_test_sample": X_test.head(10).to_dict(orient="records"),
        "y_test_sample": y_test.head(10).tolist(),
    }

    def _resolve_dir(d: str) -> str:
        if os.path.exists(d):
            return d
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cand = os.path.join(root_dir, d)
        if os.path.exists(cand):
            return cand
        cand2 = os.path.join("..", d)
        if os.path.exists(cand2):
            return cand2
        return d

    models_dir = _resolve_dir(models_dir)
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, f"{disease}_model.joblib")
    joblib.dump(artifact, model_path)
    print(f"Saved complete artifact to {model_path} ({os.path.getsize(model_path) / 1024:.1f} KB)")

    return artifact


def train_all_diseases():
    """Train and serialize all 4 disease models."""
    artifacts = {}
    for d in ["diabetes", "heart", "kidney", "cancer"]:
        artifacts[d] = train_and_evaluate_disease(d)
    return artifacts


if __name__ == "__main__":
    train_all_diseases()
