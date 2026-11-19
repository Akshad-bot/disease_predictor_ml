"""
src/predict.py
==============
Prediction engine, cluster assignment, local feature attribution,
what-if sensitivity analysis, and population comparison for all 4 diseases.
"""

import functools
import os
import joblib
import numpy as np
import pandas as pd
from src.preprocess import DISEASE_METADATA


def _resolve_model_path(models_dir: str, disease: str) -> str:
    path = os.path.join(models_dir, f"{disease}_model.joblib")
    if os.path.exists(path):
        return path
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidate = os.path.join(root_dir, models_dir, f"{disease}_model.joblib")
    if os.path.exists(candidate):
        return candidate
    candidate2 = os.path.join("..", models_dir, f"{disease}_model.joblib")
    if os.path.exists(candidate2):
        return candidate2
    return path


@functools.lru_cache(maxsize=4)
def get_model_artifact(disease: str, models_dir: str = "models") -> dict:
    """
    Loads and caches the trained artifact for the specified disease.
    """
    path = _resolve_model_path(models_dir, disease)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model artifact not found for '{disease}' at {path}. Run src/train.py first.")
    return joblib.load(path)


def prepare_input_dataframe(disease: str, input_dict: dict) -> pd.DataFrame:
    """
    Converts user input dictionary into a DataFrame matching expected columns.
    """
    meta = DISEASE_METADATA[disease]
    expected_cols = meta["numeric_cols"] + meta["categorical_cols"]
    row = {}
    for col in expected_cols:
        val = input_dict.get(col, meta["defaults"].get(col, np.nan))
        row[col] = val
    return pd.DataFrame([row])[expected_cols]


def predict_risk(disease: str, input_dict: dict, models_dir: str = "models") -> dict:
    """
    Full inference workflow for a patient input:
    1. Risk probability & predicted class
    2. Risk tier (Low, Moderate, High) with color token
    3. Unsupervised k-Means cluster assignment & cohort prevalence
    4. Top 5 contributing features for horizontal bar chart
    """
    artifact = get_model_artifact(disease, models_dir=models_dir)
    pipeline = artifact["best_pipeline"]
    meta = artifact["metadata"]
    df_input = prepare_input_dataframe(disease, input_dict)

    # 1. Supervised Prediction
    proba = pipeline.predict_proba(df_input)[0]
    p_disease = float(proba[1])
    risk_pct = round(p_disease * 100, 1)

    if risk_pct < 35.0:
        risk_level = "Low Risk"
        risk_color = "#16A34A"  # Medical Green
        risk_summary = "Clinical parameters are predominantly within normative baseline ranges."
    elif risk_pct <= 65.0:
        risk_level = "Moderate Risk"
        risk_color = "#D97706"  # Clinical Amber
        risk_summary = "Borderline or elevated markers detected. Preventative monitoring recommended."
    else:
        risk_level = "High Risk"
        risk_color = "#DC2626"  # Clinical Red
        risk_summary = "Significant clinical risk markers detected. Timely clinical consultation advised."

    # 2. Unsupervised Cluster Assignment
    cleaner = pipeline.named_steps["cleaner"]
    preprocessor = pipeline.named_steps["preprocessor"]
    kmeans_model = artifact["kmeans_model"]
    cluster_risk_map = artifact["cluster_risk_map"]

    df_cleaned = cleaner.transform(df_input)
    X_prep = preprocessor.transform(df_cleaned)
    cluster_id = int(kmeans_model.predict(X_prep)[0])
    cluster_info = cluster_risk_map.get(cluster_id, {
        "label": f"Cluster {cluster_id}",
        "positive_rate": p_disease,
        "size": 0,
    })

    # 3. Top 5 Contributing Features
    # We combine global model feature importance with patient's standardized deviation
    # from the healthy cohort mean: impact = |(val - healthy_mean) / healthy_std| * importance
    cohort_summary = artifact.get("cohort_summary", {})
    global_importances = artifact.get("feature_importances", {})

    contributions = []
    for col in meta["numeric_cols"]:
        val = float(input_dict.get(col, meta["defaults"].get(col, 0)))
        stats = cohort_summary.get(col, None)
        if stats and stats["healthy_std"] > 1e-6:
            z_dev = (val - stats["healthy_mean"]) / stats["healthy_std"]
            # Look up importance
            feat_key_match = [k for k in global_importances if col in k]
            imp = global_importances[feat_key_match[0]] if feat_key_match else 0.1
            impact_score = abs(z_dev) * imp
            # Is patient value higher or lower than healthy mean
            direction = "Elevated" if z_dev > 0 else "Low"
            contributions.append({
                "feature": col,
                "label": meta["feature_labels"].get(col, col),
                "value": val,
                "score": float(impact_score),
                "deviation": float(z_dev),
                "direction": direction,
            })

    # Sort and take top 5
    contributions.sort(key=lambda x: x["score"], reverse=True)
    top_5 = contributions[:5]

    return {
        "disease": disease,
        "disease_title": meta["title"],
        "model_name": artifact["best_model_name"],
        "risk_score": p_disease,
        "risk_percentage": risk_pct,
        "risk_level": risk_level,
        "risk_color": risk_color,
        "risk_summary": risk_summary,
        "predicted_class": int(p_disease >= 0.5),
        "cluster_id": cluster_id,
        "cluster_label": cluster_info.get("label", f"Cluster {cluster_id}"),
        "cluster_prevalence": round(cluster_info.get("positive_rate", 0) * 100, 1),
        "top_features": top_5,
        "input_data": input_dict,
    }


def what_if_simulation(disease: str, base_input: dict, modifications: dict, models_dir: str = "models") -> dict:
    """
    Evaluates risk percentage delta when user adjusts modifiable clinical parameters.
    """
    artifact = get_model_artifact(disease, models_dir=models_dir)
    pipeline = artifact["best_pipeline"]

    # Base prediction
    base_df = prepare_input_dataframe(disease, base_input)
    base_prob = float(pipeline.predict_proba(base_df)[0][1])

    # Modified prediction
    modified_input = base_input.copy()
    modified_input.update(modifications)
    mod_df = prepare_input_dataframe(disease, modified_input)
    mod_prob = float(pipeline.predict_proba(mod_df)[0][1])

    delta_prob = mod_prob - base_prob
    delta_pct = round(delta_prob * 100, 1)

    return {
        "base_risk_pct": round(base_prob * 100, 1),
        "modified_risk_pct": round(mod_prob * 100, 1),
        "delta_pct": delta_pct,
        "improved": delta_pct < 0,
        "modified_values": modifications,
    }
