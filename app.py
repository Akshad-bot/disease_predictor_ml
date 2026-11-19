"""
app.py
======
Multi-Disease Risk Prediction & Clinical Decision Support System.
Syllabus Coverage:
- UNIT I-II: Data understanding, preprocessing, EDA
- UNIT III: Supervised learning & evaluation metrics (Recall prioritized)
- UNIT IV: Unsupervised (k-Means, Hierarchical) & probabilistic (Naive Bayes)
- UNIT V: Generative AI / LLM explanation
"""

import os
import sys
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram
import streamlit as st

# Ensure project root is in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from src.preprocess import DISEASE_METADATA, load_data
from src.predict import get_model_artifact, predict_risk, what_if_simulation
from src.explain import explain_prediction

# -----------------------------------------------------------------------------
# Page Configuration & Aesthetics
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Risk Check | Clinical Decision Support",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Design Tokens & Clean CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #1F2937;
}

.stApp {
    background-color: #F7F8FA;
}

/* Header styling */
.app-header {
    margin-bottom: 1.5rem;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid #E5E7EB;
}
.app-title {
    font-size: 1.6rem;
    font-weight: 700;
    color: #0F766E;
    letter-spacing: -0.02em;
    margin: 0;
}
.app-subtitle {
    font-size: 0.9rem;
    color: #4B5563;
    margin-top: 0.25rem;
}

/* Card Styling */
.card {
    background-color: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 8px;
    padding: 1.25rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 1px 2px rgba(0,0,0,0.03);
}

.card-title {
    font-size: 1.05rem;
    font-weight: 600;
    color: #111827;
    margin-bottom: 0.75rem;
    border-bottom: 1px solid #F3F4F6;
    padding-bottom: 0.4rem;
}

/* Badges */
.badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    font-size: 0.85rem;
    font-weight: 600;
    border-radius: 9999px;
}
.badge-low {
    background-color: #DCFCE7;
    color: #166534;
    border: 1px solid #BBF7D0;
}
.badge-moderate {
    background-color: #FEF3C7;
    color: #92400E;
    border: 1px solid #FDE68A;
}
.badge-high {
    background-color: #FEE2E2;
    color: #991B1B;
    border: 1px solid #FECACA;
}

/* Sidebar clean styling */
section[data-testid="stSidebar"] {
    background-color: #FFFFFF;
    border-right: 1px solid #E5E7EB;
}

/* Minimal Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}
.stTabs [data-baseweb="tab"] {
    height: 42px;
    padding-left: 18px;
    padding-right: 18px;
    background-color: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 6px;
    font-weight: 500;
    color: #4B5563;
}
.stTabs [aria-selected="true"] {
    background-color: #0F766E !important;
    color: #FFFFFF !important;
    border-color: #0F766E !important;
}

/* Footer */
.footer-text {
    font-size: 0.8rem;
    color: #6B7280;
    text-align: center;
    padding: 2.5rem 0 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# Matplotlib global styling to match dashboard palette
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#E5E7EB'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.color'] = '#F3F4F6'
PRIMARY_TEAL = '#0F766E'
SECONDARY_SLATE = '#475569'
RISK_COLORS = {'Low Risk': '#16A34A', 'Moderate Risk': '#D97706', 'High Risk': '#DC2626'}

# -----------------------------------------------------------------------------
# Sidebar: Disease Selection & Model Info
# -----------------------------------------------------------------------------
st.sidebar.markdown("""
<div style="padding-bottom: 0.5rem; margin-bottom: 0.75rem; border-bottom: 1px solid #E5E7EB;">
    <h3 style="margin: 0; font-size: 1.2rem; font-weight: 700; color: #0F766E;">Risk Check</h3>
    <p style="margin: 0; font-size: 0.82rem; color: #6B7280;">Clinical Risk Stratification</p>
</div>
""", unsafe_allow_html=True)

DISEASE_KEYS = {
    "Diabetes": "diabetes",
    "Heart Disease": "heart",
    "Chronic Kidney Disease": "kidney",
    "Breast Cancer": "cancer",
}

selected_disease_label = st.sidebar.radio(
    "Select Target Disease",
    options=list(DISEASE_KEYS.keys()),
    index=0,
)
current_disease = DISEASE_KEYS[selected_disease_label]
artifact = get_model_artifact(current_disease)
meta = DISEASE_METADATA[current_disease]
if not meta.get("defaults"):
    meta["defaults"] = artifact["metadata"].get("defaults", {})
if not meta.get("ranges"):
    meta["ranges"] = artifact["metadata"].get("ranges", {})


st.sidebar.markdown("---")
st.sidebar.markdown(f"**Primary Model:** `{artifact['best_model_name']}`")
st.sidebar.markdown(f"**Optimization Metric:** `Recall (Sensitivity)`")
test_metrics = artifact['metrics_df']
best_row = test_metrics[test_metrics['Model'] == artifact['best_model_name']].iloc[0]
st.sidebar.markdown(
    f"- Test Recall: **{best_row['Recall']:.1%}**\n"
    f"- Test Accuracy: **{best_row['Accuracy']:.1%}**\n"
    f"- ROC-AUC: **{best_row['ROC-AUC']:.3f}**"
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """<div style="font-size: 0.78rem; color: #6B7280; line-height: 1.4;">
    <strong>College ML Project</strong><br>
    Units I-V Comprehensive Implementation.<br>
    <em>Educational clinical decision support only.</em>
    </div>""",
    unsafe_allow_html=True
)

# -----------------------------------------------------------------------------
# Main Header
# -----------------------------------------------------------------------------
st.markdown(f"""
<div class="app-header">
    <h1 class="app-title">{meta['title']} Assessment</h1>
    <p class="app-subtitle">Evidence-based predictive risk scoring, cluster stratification, and generative plain-language explanations.</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Main Navigation Tabs
# -----------------------------------------------------------------------------
tab_predict, tab_insights, tab_models, tab_about = st.tabs([
    "Predict & Stratify",
    "Data Insights (EDA)",
    "Model Benchmarks",
    "Syllabus & About",
])


# =============================================================================
# TAB 1: PREDICT & STRATIFY
# =============================================================================
with tab_predict:
    col_input, col_result = st.columns([1.05, 1.15], gap="large")

    # -------------------------------------------------------------------------
    # LEFT COLUMN: INPUT FORM
    # -------------------------------------------------------------------------
    with col_input:
        st.markdown('<div class="card-title">Patient Clinical Parameters</div>', unsafe_allow_html=True)

        user_inputs = {}

        if current_disease == "diabetes":
            st.caption("Enter patient demographic and metabolic biomarkers.")
            c1, c2 = st.columns(2)
            with c1:
                user_inputs["Age"] = st.number_input("Age (years)", min_value=21, max_value=90, value=int(meta["defaults"]["Age"]))
                user_inputs["Pregnancies"] = st.number_input("Pregnancies", min_value=0, max_value=17, value=int(meta["defaults"]["Pregnancies"]))
                user_inputs["Glucose"] = st.number_input("Fasting Glucose (mg/dL)", min_value=50.0, max_value=250.0, value=float(meta["defaults"]["Glucose"]), step=1.0)
                user_inputs["BloodPressure"] = st.number_input("Diastolic BP (mm Hg)", min_value=40.0, max_value=140.0, value=float(meta["defaults"]["BloodPressure"]), step=1.0)
            with c2:
                user_inputs["BMI"] = st.number_input("BMI (kg/m²)", min_value=15.0, max_value=65.0, value=float(meta["defaults"]["BMI"]), step=0.1)
                user_inputs["Insulin"] = st.number_input("2-Hour Insulin (μU/mL)", min_value=10.0, max_value=850.0, value=float(meta["defaults"]["Insulin"]), step=1.0)
                user_inputs["SkinThickness"] = st.number_input("Skin Thickness (mm)", min_value=5.0, max_value=100.0, value=float(meta["defaults"]["SkinThickness"]), step=1.0)
                user_inputs["DiabetesPedigreeFunction"] = st.number_input("Pedigree Genetic Score", min_value=0.05, max_value=2.50, value=float(meta["defaults"]["DiabetesPedigreeFunction"]), step=0.01)

        elif current_disease == "heart":
            st.caption("Enter cardiovascular hemodynamics and exercise test results.")
            c1, c2 = st.columns(2)
            with c1:
                user_inputs["Age"] = st.number_input("Age (years)", min_value=28, max_value=80, value=int(meta["defaults"]["Age"]))
                sex_options = meta["category_options"]["Sex"]
                selected_sex = st.selectbox("Biological Sex", options=list(sex_options.keys()), format_func=lambda x: sex_options[x])
                user_inputs["Sex"] = selected_sex

                cp_options = meta["category_options"]["ChestPainType"]
                selected_cp = st.selectbox("Chest Pain Type", options=list(cp_options.keys()), format_func=lambda x: cp_options[x])
                user_inputs["ChestPainType"] = selected_cp

                user_inputs["RestingBP"] = st.number_input("Resting BP (mm Hg)", min_value=80.0, max_value=200.0, value=float(meta["defaults"]["RestingBP"]), step=1.0)
                user_inputs["Cholesterol"] = st.number_input("Serum Cholesterol (mg/dL)", min_value=100.0, max_value=600.0, value=float(meta["defaults"]["Cholesterol"]), step=1.0)

            with c2:
                fbs_options = meta["category_options"]["FastingBS"]
                selected_fbs = st.selectbox("Fasting Blood Sugar", options=list(fbs_options.keys()), format_func=lambda x: fbs_options[x])
                user_inputs["FastingBS"] = selected_fbs

                ecg_options = meta["category_options"]["RestingECG"]
                selected_ecg = st.selectbox("Resting ECG", options=list(ecg_options.keys()), format_func=lambda x: ecg_options[x])
                user_inputs["RestingECG"] = selected_ecg

                user_inputs["MaxHR"] = st.number_input("Max Heart Rate (bpm)", min_value=60.0, max_value=210.0, value=float(meta["defaults"]["MaxHR"]), step=1.0)

                ang_options = meta["category_options"]["ExerciseAngina"]
                selected_ang = st.selectbox("Exercise Angina", options=list(ang_options.keys()), format_func=lambda x: ang_options[x])
                user_inputs["ExerciseAngina"] = selected_ang

                user_inputs["Oldpeak"] = st.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=6.5, value=float(meta["defaults"]["Oldpeak"]), step=0.1)

                slope_options = meta["category_options"]["ST_Slope"]
                selected_slope = st.selectbox("Peak Exercise ST Slope", options=list(slope_options.keys()), format_func=lambda x: slope_options[x])
                user_inputs["ST_Slope"] = selected_slope

        elif current_disease == "kidney":
            st.caption("Enter renal panel, urinalysis, and clinical history parameters.")
            tab_k1, tab_k2, tab_k3 = st.tabs(["Biochemistry", "Urinalysis", "History & Vitals"])
            with tab_k1:
                c1, c2 = st.columns(2)
                with c1:
                    user_inputs["age"] = st.number_input("Age (years)", 1.0, 95.0, float(meta["defaults"]["age"]))
                    user_inputs["bgr"] = st.number_input("Blood Glucose Random (mg/dL)", 50.0, 500.0, float(meta["defaults"]["bgr"]))
                    user_inputs["bu"] = st.number_input("Blood Urea (mg/dL)", 10.0, 390.0, float(meta["defaults"]["bu"]))
                    user_inputs["sc"] = st.number_input("Serum Creatinine (mg/dL)", 0.4, 20.0, float(meta["defaults"]["sc"]), step=0.1)
                with c2:
                    user_inputs["sod"] = st.number_input("Sodium (mEq/L)", 100.0, 165.0, float(meta["defaults"]["sod"]))
                    user_inputs["pot"] = st.number_input("Potassium (mEq/L)", 2.0, 9.0, float(meta["defaults"]["pot"]), step=0.1)
                    user_inputs["hemo"] = st.number_input("Hemoglobin (g/dL)", 3.0, 18.0, float(meta["defaults"]["hemo"]), step=0.1)
                    user_inputs["pcv"] = st.number_input("Packed Cell Volume (%)", 10.0, 60.0, float(meta["defaults"]["pcv"]))
                    user_inputs["wc"] = st.number_input("White Blood Cells (/μL)", 2200.0, 26000.0, float(meta["defaults"]["wc"]), step=100.0)
                    user_inputs["rc"] = st.number_input("Red Blood Cells (M/μL)", 2.0, 8.0, float(meta["defaults"]["rc"]), step=0.1)

            with tab_k2:
                c1, c2 = st.columns(2)
                with c1:
                    user_inputs["bp"] = st.number_input("Blood Pressure (mm Hg)", 50.0, 180.0, float(meta["defaults"]["bp"]))
                    user_inputs["sg"] = st.selectbox("Specific Gravity", options=[1.005, 1.010, 1.015, 1.020, 1.025], index=3)
                    user_inputs["al"] = st.selectbox("Albumin Level (0-5 scale)", options=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], index=0)
                    user_inputs["su"] = st.selectbox("Sugar Level (0-5 scale)", options=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], index=0)
                with c2:
                    user_inputs["rbc"] = st.selectbox("Red Blood Cells in Urine", ["normal", "abnormal"], format_func=lambda x: x.capitalize())
                    user_inputs["pc"] = st.selectbox("Pus Cells in Urine", ["normal", "abnormal"], format_func=lambda x: x.capitalize())
                    user_inputs["pcc"] = st.selectbox("Pus Cell Clumps", ["notpresent", "present"], format_func=lambda x: "Not Present" if x=="notpresent" else "Present")
                    user_inputs["ba"] = st.selectbox("Bacteria in Urine", ["notpresent", "present"], format_func=lambda x: "Not Present" if x=="notpresent" else "Present")

            with tab_k3:
                c1, c2 = st.columns(2)
                with c1:
                    user_inputs["htn"] = st.selectbox("Hypertension", ["no", "yes"], format_func=lambda x: x.capitalize())
                    user_inputs["dm"] = st.selectbox("Diabetes Mellitus", ["no", "yes"], format_func=lambda x: x.capitalize())
                    user_inputs["cad"] = st.selectbox("Coronary Artery Disease", ["no", "yes"], format_func=lambda x: x.capitalize())
                with c2:
                    user_inputs["appet"] = st.selectbox("Appetite", ["good", "poor"], format_func=lambda x: x.capitalize())
                    user_inputs["pe"] = st.selectbox("Pedal Edema", ["no", "yes"], format_func=lambda x: x.capitalize())
                    user_inputs["ane"] = st.selectbox("Anemia", ["no", "yes"], format_func=lambda x: x.capitalize())

        elif current_disease == "cancer":
            st.caption("Enter Fine Needle Aspirate (FNA) cell nuclei biometric characteristics.")
            preset = st.radio("Load Parameter Preset", ["Dataset Median (Baseline)", "Typical Malignant Profile", "Custom"], horizontal=True)

            # Define typical presets
            if preset == "Typical Malignant Profile":
                def_mod = {
                    "mean radius": 18.0, "mean texture": 22.0, "mean perimeter": 120.0,
                    "mean area": 1050.0, "mean concave points": 0.09, "worst radius": 22.0,
                    "worst perimeter": 150.0, "worst area": 1500.0, "worst concave points": 0.18
                }
            else:
                def_mod = {}

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Mean Characteristics**")
                for feat in ["mean radius", "mean texture", "mean perimeter", "mean area", "mean concave points"]:
                    default_v = def_mod.get(feat, meta["defaults"].get(feat, artifact["metadata"]["defaults"].get(feat, 14.0)))
                    min_v, max_v, step_v = meta["ranges"].get(feat, artifact["metadata"]["ranges"].get(feat, (0.0, 100.0, 0.1)))
                    user_inputs[feat] = st.number_input(meta["feature_labels"].get(feat, feat), float(min_v), float(max_v), float(default_v), step=float(step_v))
            with c2:
                st.markdown("**Extreme / Worst Characteristics**")
                for feat in ["worst radius", "worst texture", "worst perimeter", "worst area", "worst concave points"]:
                    default_v = def_mod.get(feat, meta["defaults"].get(feat, artifact["metadata"]["defaults"].get(feat, 15.0)))
                    min_v, max_v, step_v = meta["ranges"].get(feat, artifact["metadata"]["ranges"].get(feat, (0.0, 100.0, 0.1)))
                    user_inputs[feat] = st.number_input(meta["feature_labels"].get(feat, feat), float(min_v), float(max_v), float(default_v), step=float(step_v))

            # Fill remaining features with default medians to keep form clean
            for col in meta["numeric_cols"]:
                if col not in user_inputs:
                    user_inputs[col] = meta["defaults"].get(col, artifact["metadata"]["defaults"].get(col, 0.0))

    # -------------------------------------------------------------------------
    # RIGHT COLUMN: RESULTS, WHAT-IF, & CLINICAL EXPLANATION
    # -------------------------------------------------------------------------
    with col_result:
        # Run prediction
        res = predict_risk(current_disease, user_inputs)
        risk_pct = res["risk_percentage"]
        risk_level = res["risk_level"]
        risk_color = res["risk_color"]

        # Risk Score Card
        badge_class = "badge-low" if risk_level == "Low Risk" else ("badge-moderate" if risk_level == "Moderate Risk" else "badge-high")
        st.markdown(f"""
        <div class="card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                <span style="font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: #6B7280;">Predicted Clinical Risk</span>
                <span class="badge {badge_class}">{risk_level}</span>
            </div>
            <div style="display: flex; align-items: baseline; gap: 0.75rem;">
                <span style="font-size: 2.4rem; font-weight: 700; color: {risk_color}; letter-spacing: -0.03em;">{risk_pct:.1f}%</span>
                <span style="font-size: 0.9rem; color: #6B7280;">probability of disease</span>
            </div>
            <div style="margin: 0.75rem 0 0.5rem 0; background-color: #E5E7EB; border-radius: 9999px; height: 8px; overflow: hidden;">
                <div style="background-color: {risk_color}; width: {min(100.0, max(2.0, risk_pct))}%; height: 100%;"></div>
            </div>
            <p style="font-size: 0.85rem; color: #4B5563; margin-top: 0.5rem; margin-bottom: 0;">{res['risk_summary']}</p>
        </div>
        """, unsafe_allow_html=True)

        # Unsupervised Cluster Badge Card
        st.markdown(f"""
        <div class="card" style="padding: 1rem 1.25rem;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <div style="font-size: 0.78rem; font-weight: 600; color: #6B7280; text-transform: uppercase;">Unsupervised Stratification (k-Means)</div>
                    <div style="font-size: 1.05rem; font-weight: 600; color: #111827; margin-top: 0.2rem;">{res['cluster_label']}</div>
                </div>
                <div style="text-align: right;">
                    <span style="font-size: 1.2rem; font-weight: 700; color: #0F766E;">{res['cluster_prevalence']}%</span>
                    <div style="font-size: 0.75rem; color: #6B7280;">cohort prevalence</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Top 5 Contributing Features Chart
        st.markdown('<div class="card-title">Top Contributing Clinical Factors</div>', unsafe_allow_html=True)
        top_f = res["top_features"]
        if top_f:
            feat_names = [f["label"] for f in reversed(top_f)]
            feat_scores = [f["score"] for f in reversed(top_f)]
            feat_colors = ['#DC2626' if f["direction"] == "Elevated" else '#0F766E' for f in reversed(top_f)]

            fig_bar, ax_bar = plt.subplots(figsize=(6.5, 2.6))
            bars = ax_bar.barh(feat_names, feat_scores, color=feat_colors, height=0.55)
            ax_bar.set_xlabel('Relative Impact Score', fontsize=9, color='#6B7280')
            ax_bar.tick_params(axis='both', which='major', labelsize=8.5)
            ax_bar.grid(axis='x', linestyle='--', alpha=0.5)
            ax_bar.spines['top'].set_visible(False)
            ax_bar.spines['right'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig_bar)
            plt.close(fig_bar)

        # ---------------------------------------------------------------------
        # WHAT-IF SENSITIVITY SIMULATION PANEL
        # ---------------------------------------------------------------------
        st.markdown('<div class="card-title" style="margin-top: 1.5rem;">What-If Clinical Sensitivity Simulator</div>', unsafe_allow_html=True)
        st.caption("Adjust key modifiable lifestyle or metabolic biomarkers to evaluate live impact on risk.")

        mod_cols = meta.get("modifiable_cols", [])[:3]
        sim_adjustments = {}

        cols_sliders = st.columns(len(mod_cols))
        for idx, m_col in enumerate(mod_cols):
            with cols_sliders[idx]:
                min_v, max_v, step_v = meta["ranges"].get(m_col, (0.0, 100.0, 1.0))
                current_v = float(user_inputs.get(m_col, meta["defaults"].get(m_col, 50.0)))
                # Constrain within bounds
                current_v = max(float(min_v), min(float(max_v), current_v))
                short_label = meta["feature_labels"].get(m_col, m_col).split("(")[0].strip()
                sim_adjustments[m_col] = st.slider(
                    f"{short_label}",
                    min_value=float(min_v),
                    max_value=float(max_v),
                    value=current_v,
                    step=float(step_v),
                    key=f"sim_{m_col}",
                )

        # Compute simulation
        sim_res = what_if_simulation(current_disease, user_inputs, sim_adjustments)
        new_risk = sim_res["modified_risk_pct"]
        delta = sim_res["delta_pct"]

        delta_color = "#16A34A" if delta <= 0 else "#DC2626"
        delta_sign = "" if delta <= 0 else "+"
        st.markdown(f"""
        <div style="background-color: #F8FAFC; border: 1px dashed #CBD5E1; border-radius: 6px; padding: 0.75rem 1rem; margin-top: 0.5rem; display: flex; justify-content: space-between; align-items: center;">
            <span style="font-size: 0.88rem; color: #475569;">Simulated Risk Probability: <strong>{new_risk:.1f}%</strong></span>
            <span style="font-size: 0.88rem; font-weight: 700; color: {delta_color};">{delta_sign}{delta:.1f}% risk shift</span>
        </div>
        """, unsafe_allow_html=True)

        # ---------------------------------------------------------------------
        # COMPARE TO DATASET VIEW
        # ---------------------------------------------------------------------
        st.markdown('<div class="card-title" style="margin-top: 1.5rem;">Population Baseline Comparison</div>', unsafe_allow_html=True)
        st.caption("Visualizing patient values alongside healthy vs diseased cohort distributions.")

        key_compare_cols = meta.get("modifiable_cols", [])[:2]
        if key_compare_cols and "cohort_summary" in artifact:
            fig_comp, ax_comp = plt.subplots(1, len(key_compare_cols), figsize=(7, 2.2))
            if len(key_compare_cols) == 1:
                ax_comp = [ax_comp]

            for idx, c_col in enumerate(key_compare_cols):
                stats = artifact["cohort_summary"].get(c_col)
                if stats:
                    patient_val = float(user_inputs.get(c_col, 0))
                    # Plot cohort box representation
                    h_med = stats["healthy_median"]
                    d_med = stats["disease_median"]

                    ax_comp[idx].scatter([h_med], [0], color='#0F766E', s=90, label='Healthy Cohort Median', zorder=3)
                    ax_comp[idx].scatter([d_med], [1], color='#DC2626', s=90, label='Diseased Cohort Median', zorder=3)
                    ax_comp[idx].scatter([patient_val], [0.5], color='#D97706', s=140, marker='*', label='Patient Value', zorder=4)

                    ax_comp[idx].set_yticks([0, 0.5, 1])
                    ax_comp[idx].set_yticklabels(['Healthy', 'Patient', 'Diseased'], fontsize=8)
                    ax_comp[idx].set_title(meta['feature_labels'].get(c_col, c_col).split("(")[0], fontsize=9)
                    ax_comp[idx].grid(axis='x', linestyle='--', alpha=0.4)
                    ax_comp[idx].spines['top'].set_visible(False)
                    ax_comp[idx].spines['right'].set_visible(False)

            plt.tight_layout()
            st.pyplot(fig_comp)
            plt.close(fig_comp)

        # ---------------------------------------------------------------------
        # GENERATIVE AI CLINICAL EXPLANATION & DOWNLOAD
        # ---------------------------------------------------------------------
        st.markdown('<div class="card-title" style="margin-top: 1.5rem;">Clinical Explanation & Documentation</div>', unsafe_allow_html=True)

        if st.button("Generate Plain-Language Explanation", use_container_width=True, type="primary"):
            with st.spinner("Synthesizing patient metrics with clinical knowledge engine..."):
                exp_res = explain_prediction(current_disease, user_inputs, res)
                st.session_state["latest_explanation"] = exp_res

        if "latest_explanation" in st.session_state:
            exp_data = st.session_state["latest_explanation"]
            st.markdown(f"""
            <div style="background-color: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 8px; padding: 1.25rem; margin-top: 1rem;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem; border-bottom: 1px solid #F3F4F6; padding-bottom: 0.4rem;">
                    <span style="font-size: 0.85rem; font-weight: 600; color: #0F766E;">AI Clinical Synthesis</span>
                    <span style="font-size: 0.75rem; color: #6B7280;">{exp_data.get('source', 'Clinical Engine')}</span>
                </div>
            """, unsafe_allow_html=True)
            st.markdown(exp_data["explanation"])
            st.markdown("</div>", unsafe_allow_html=True)

            # Download Clinical Summary Report
            report_text = f"""========================================================================
RISK CHECK - CLINICAL DECISION SUPPORT SUMMARY REPORT
========================================================================
Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Disease Evaluated: {meta['title']}
Primary Model: {res['model_name']} (Recall Optimized)

PATIENT RISK STRATIFICATION
------------------------------------------------------------------------
- Estimated Disease Risk: {risk_pct:.1f}%
- Risk Classification: {risk_level}
- Unsupervised Cluster: {res['cluster_label']} ({res['cluster_prevalence']}% historical prevalence)

TOP CONTRIBUTING CLINICAL FACTORS
------------------------------------------------------------------------
""" + "\n".join([f"- {f['label']}: {f['value']} ({f['direction']}, impact score: {f['score']:.2f})" for f in res['top_features']]) + f"""

CLINICAL EXPLANATION & RECOMMENDATIONS
------------------------------------------------------------------------
{exp_data['explanation']}

========================================================================
DISCLAIMER: Educational machine learning project, not a medical diagnosis.
========================================================================
"""
            st.download_button(
                label="Download Clinical Summary Report (.txt)",
                data=report_text,
                file_name=f"{current_disease}_risk_summary_{datetime.date.today()}.txt",
                mime="text/plain",
                use_container_width=True,
            )


# =============================================================================
# TAB 2: DATA INSIGHTS (EDA)
# =============================================================================
with tab_insights:
    st.markdown(f"### Exploratory Data Analysis: {meta['title']}")
    st.caption("UNIT I-II: Univariate distributions, class balance, correlation heatmaps, boxplots, and 2D PCA projection.")

    X_raw, y_raw = load_data(current_disease)

    # Row 1: Class Balance & Distributions
    c_eda1, c_eda2 = st.columns([1, 1.2])
    with c_eda1:
        st.markdown("**Outcome Class Balance**")
        fig_cb, ax_cb = plt.subplots(figsize=(5, 3.2))
        sns.countplot(x=y_raw, palette=['#0F766E', '#DC2626'], ax=ax_cb)
        ax_cb.set_xticklabels(meta["target_names"], fontsize=8.5)
        ax_cb.set_ylabel('Patient Count', fontsize=9)
        ax_cb.set_xlabel('')
        ax_cb.grid(axis='y', linestyle='--', alpha=0.5)
        ax_cb.spines['top'].set_visible(False)
        ax_cb.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig_cb)
        plt.close(fig_cb)

    with c_eda2:
        st.markdown("**Primary Biomarker Distribution**")
        primary_feat = meta["numeric_cols"][0] if meta["numeric_cols"] else "Glucose"
        fig_dist, ax_dist = plt.subplots(figsize=(6, 3.2))
        sns.histplot(X_raw[primary_feat].dropna(), kde=True, color='#0F766E', ax=ax_dist)
        ax_dist.set_title(meta["feature_labels"].get(primary_feat, primary_feat), fontsize=9)
        ax_dist.grid(axis='y', linestyle='--', alpha=0.5)
        ax_dist.spines['top'].set_visible(False)
        ax_dist.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig_dist)
        plt.close(fig_dist)

    # Row 2: Correlation Heatmap & 2D PCA
    c_eda3, c_eda4 = st.columns([1.2, 1])
    with c_eda3:
        st.markdown("**Correlation Matrix Heatmap**")
        fig_heat, ax_heat = plt.subplots(figsize=(7, 4.5))
        num_sub = X_raw[meta["numeric_cols"][:8]].dropna()
        sns.heatmap(num_sub.corr(), annot=True, fmt='.2f', cmap='Blues', ax=ax_heat, cbar=True, annot_kws={'size': 7.5})
        plt.xticks(fontsize=8, rotation=45, ha='right')
        plt.yticks(fontsize=8)
        plt.tight_layout()
        st.pyplot(fig_heat)
        plt.close(fig_heat)

    with c_eda4:
        st.markdown("**2D PCA Feature Space Projection**")
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer

        X_imputed = SimpleImputer(strategy='median').fit_transform(X_raw.select_dtypes(include=[np.number]))
        X_scaled = StandardScaler().fit_transform(X_imputed)
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_scaled)

        fig_pca, ax_pca = plt.subplots(figsize=(6, 4.5))
        scatter = ax_pca.scatter(X_pca[:, 0], X_pca[:, 1], c=y_raw, cmap='coolwarm', alpha=0.75, edgecolors='k', s=30)
        ax_pca.set_title(f'PCA Variance Explained: {pca.explained_variance_ratio_.sum():.1%}', fontsize=9.5)
        ax_pca.set_xlabel('Principal Component 1', fontsize=9)
        ax_pca.set_ylabel('Principal Component 2', fontsize=9)
        ax_pca.grid(True, linestyle='--', alpha=0.4)
        cbar = plt.colorbar(scatter, ax=ax_pca)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(['Healthy', 'Disease'])
        plt.tight_layout()
        st.pyplot(fig_pca)
        plt.close(fig_pca)


# =============================================================================
# TAB 3: MODEL BENCHMARKS
# =============================================================================
with tab_models:
    st.markdown(f"### Model Evaluation & Clinical Benchmark: {meta['title']}")
    st.caption("UNIT III & IV: Cross-validation, test metrics, Recall prioritization, ROC curves, kNN tuning, and clustering.")

    # 1. Comparison Table
    st.markdown('<div class="card-title">Model Comparison Table (Held-Out Test Set)</div>', unsafe_allow_html=True)
    metrics_display = artifact["metrics_df"].copy()
    # Format percentages
    for col in ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "CV Recall"]:
        if col in metrics_display.columns:
            metrics_display[col] = metrics_display[col].apply(lambda x: f"{x:.1%}" if isinstance(x, (float, int)) else x)
    st.dataframe(metrics_display, use_container_width=True, hide_index=True)

    st.info(f"**Medical Model Selection Rationale:** {artifact['selection_reason']}")

    # 2. ROC Curves & Confusion Matrix
    c_m1, c_m2 = st.columns([1.1, 1])
    with c_m1:
        st.markdown("**Multi-Model ROC Curves**")
        fig_roc, ax_roc = plt.subplots(figsize=(6.5, 4.2))
        for m_name, r_data in artifact["roc_curves"].items():
            ax_roc.plot(r_data["fpr"], r_data["tpr"], label=f"{m_name} (AUC={r_data['auc']:.2f})", linewidth=1.8)
        ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.4)
        ax_roc.set_xlabel('False Positive Rate', fontsize=9)
        ax_roc.set_ylabel('True Positive Rate (Recall)', fontsize=9)
        ax_roc.legend(fontsize=8, loc='lower right')
        ax_roc.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        st.pyplot(fig_roc)
        plt.close(fig_roc)

    with c_m2:
        st.markdown(f"**Confusion Matrix ({artifact['best_model_name']})**")
        cm = np.array(artifact["confusion_matrices"][artifact["best_model_name"]])
        fig_cm, ax_cm = plt.subplots(figsize=(5.5, 4.2))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                    xticklabels=['Pred Healthy', 'Pred Disease'],
                    yticklabels=['Actual Healthy', 'Actual Disease'])
        ax_cm.set_ylabel('Ground Truth', fontsize=9)
        ax_cm.set_xlabel('Predicted Label', fontsize=9)
        plt.tight_layout()
        st.pyplot(fig_cm)
        plt.close(fig_cm)

    st.markdown("---")

    # 3. kNN K-Tuning & Clustering Visualizations
    c_m3, c_m4 = st.columns(2)
    with c_m3:
        st.markdown("**kNN Accuracy vs K Hyperparameter Curve**")
        k_scores = artifact["knn_k_scores"]
        fig_k, ax_k = plt.subplots(figsize=(6, 3.2))
        ax_k.plot(list(k_scores.keys()), list(k_scores.values()), marker='o', color='#0F766E', linewidth=1.8)
        ax_k.axvline(artifact["best_k"], color='#DC2626', linestyle='--', label=f'Best K={artifact["best_k"]}')
        ax_k.set_xlabel('K Neighbors', fontsize=9)
        ax_k.set_ylabel('5-Fold CV Accuracy', fontsize=9)
        ax_k.legend(fontsize=8.5)
        ax_k.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        st.pyplot(fig_k)
        plt.close(fig_k)

    with c_m4:
        st.markdown("**k-Means Inertia (Elbow) & Silhouette Analysis**")
        km_eval = artifact["kmeans_eval"]
        fig_km, ax_km1 = plt.subplots(figsize=(6, 3.2))
        ax_km1.plot(km_eval["k_range"], km_eval["inertia"], marker='o', color='#0F766E', label='Inertia')
        ax_km1.set_xlabel('Number of Clusters (K)', fontsize=9)
        ax_km1.set_ylabel('Inertia (Sum of Squares)', fontsize=9, color='#0F766E')
        ax_km2 = ax_km1.twinx()
        ax_km2.plot(km_eval["k_range"], km_eval["silhouette"], marker='s', color='#D97706', linestyle='--', label='Silhouette')
        ax_km2.set_ylabel('Silhouette Score', fontsize=9, color='#D97706')
        ax_km2.grid(False)
        plt.tight_layout()
        st.pyplot(fig_km)
        plt.close(fig_km)

    # 4. Hierarchical Dendrogram
    st.markdown("**Hierarchical Clustering Dendrogram (Ward's Linkage)**")
    fig_dendro, ax_dendro = plt.subplots(figsize=(12, 3.5))
    dendrogram(artifact["linkage_matrix"], ax=ax_dendro, truncate_mode='lastp', p=30, leaf_rotation=90, leaf_font_size=8)
    ax_dendro.set_ylabel('Distance (Ward Linkage)', fontsize=9)
    ax_dendro.set_xlabel('Patient Cluster Merges', fontsize=9)
    plt.tight_layout()
    st.pyplot(fig_dendro)
    plt.close(fig_dendro)


# =============================================================================
# TAB 4: ABOUT & SYLLABUS MAPPING
# =============================================================================
with tab_about:
    st.markdown("### Coursework & Technical Architecture")
    st.caption("Complete mapping of syllabus units to code modules, notebooks, and algorithms.")

    st.markdown("""
| Unit | Syllabus Topic | Implementation in Project | File Reference |
| :--- | :--- | :--- | :--- |
| **UNIT I** | Data Understanding, Cleaning & Quality Fixes | - Zeros to NaN in Diabetes (Glucose, BP, Skin, Insulin, BMI)<br>- Heart: Cholesterol/RestingBP zeros to NaN, Oldpeak negative clipping<br>- Kidney: '?' and whitespace cleaning, coercion of pcv/wc/rc, id drop<br>- Breast cancer: target inversion for clinical consistency | `src/preprocess.py`<br>`notebooks/01_eda_preprocessing.ipynb` |
| **UNIT II** | Preprocessing Pipelines & EDA | - Scikit-Learn `ColumnTransformer` with `SimpleImputer`, `StandardScaler`, `OneHotEncoder`<br>- Univariate distributions, Bivariate boxplots, Correlation heatmaps, 2D PCA projection | `src/preprocess.py`<br>`notebooks/01_eda_preprocessing.ipynb` |
| **UNIT III** | Supervised Learning & Evaluation Metrics | - 5 Algorithms: kNN (with K tuning curve), Logistic Regression, Decision Tree, Random Forest, Gaussian Naive Bayes<br>- 5-Fold Stratified Cross Validation<br>- Metrics: Accuracy, Precision, Recall, F1, ROC-AUC<br>- Clinical Recall prioritization | `src/train.py`<br>`notebooks/02_models.ipynb`<br>Streamlit Models Tab |
| **UNIT IV** | Unsupervised & Probabilistic Learning | - k-Means Clustering: Elbow & Silhouette evaluation ($K=2..6$)<br>- Empirical cluster risk labeling (Low/Moderate/High)<br>- Hierarchical Clustering: Ward's Linkage Dendrogram<br>- Gaussian Naive Bayes: Probabilistic class posteriors | `src/train.py`<br>`src/predict.py`<br>`notebooks/02_models.ipynb` |
| **UNIT V** | Generative AI & LLM Explanation | - Plain-language clinical explanation via Google Generative AI (Gemini API)<br>- Graceful offline template fallback engine ensuring system never crashes | `src/explain.py`<br>Streamlit Predict Tab |
""")

    st.markdown("""
### Methodology & Engineering Notes
1. **Data Leakage Prevention:** Preprocessing pipelines fit solely on training partitions. Imputation medians and scaling parameters are learned strictly during training and applied identically at inference.
2. **Clinical Decision Prioritization:** In medical screening, **False Negatives** can be life-threatening, while False Positives can be cleared via routine confirmatory testing. Models were chosen based on held-out **Recall (Sensitivity)**.
3. **Resilience & Fallbacks:** The Generative AI explanation gracefully switches between live Gemini LLM generation and a rule-based clinical engine if `GEMINI_API_KEY` is omitted or offline.
""")

# -----------------------------------------------------------------------------
# Footer Disclaimer
# -----------------------------------------------------------------------------
st.markdown("""
<div class="footer-text">
    Educational project, not a medical diagnosis. Consult qualified medical practitioners for diagnostic evaluations.
</div>
""", unsafe_allow_html=True)
