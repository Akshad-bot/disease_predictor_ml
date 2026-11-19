"""
src/explain.py
==============
UNIT V: Generative AI / LLM Explanation of Risk Prediction.

Uses Google Generative AI (Gemini) when GEMINI_API_KEY is available in the environment.
Falls back seamlessly to a high-quality clinical template-based explanation
if the key is absent or network fails, ensuring the system never breaks.
"""

import os
import google.generativeai as genai


def generate_template_explanation(disease_title: str, prediction_result: dict, input_dict: dict) -> str:
    """
    High-quality deterministic fallback explanation constructed from clinical logic.
    """
    risk_level = prediction_result["risk_level"]
    risk_pct = prediction_result["risk_percentage"]
    cluster_label = prediction_result["cluster_label"]
    cluster_prev = prediction_result["cluster_prevalence"]
    top_features = prediction_result.get("top_features", [])

    # Formulate drivers
    drivers_md = ""
    for f in top_features[:3]:
        name = f["label"]
        val = f["value"]
        direction = f["direction"]
        drivers_md += f"- **{name} ({val})**: Currently marked as *{direction.lower()}* relative to baseline healthy controls, contributing heavily to the risk score.\n"

    # Actionable suggestions per disease
    lifestyle_tips = {
        "Diabetes Risk": (
            "Focus on complex carbohydrates with low glycemic index, maintain consistent physical activity "
            "(aim for 150 minutes of moderate aerobic exercise weekly), and monitor fasting blood glucose levels regularly."
        ),
        "Heart Disease Risk": (
            "Adopt a heart-healthy Mediterranean dietary pattern low in saturated fats and sodium, engage in regular cardiovascular exercise, "
            "and track blood pressure and lipid panels consistently with your primary care provider."
        ),
        "Chronic Kidney Disease Risk": (
            "Maintain optimal blood pressure (<130/80 mm Hg), stay adequately hydrated, limit unnecessary NSAID analgesic use, "
            "and manage blood sugar levels diligently."
        ),
        "Breast Cancer Risk": (
            "Schedule routine age-appropriate clinical mammograms and breast exams, discuss family history with a genetic counselor if indicated, "
            "and maintain physical wellness and balanced nutrition."
        ),
    }

    advice = lifestyle_tips.get(
        disease_title,
        "Consult your healthcare professional for tailored preventative screening and lifestyle guidance."
    )

    return f"""### Clinical Overview
Based on your input parameters, the predictive model estimates your **{disease_title}** risk at **{risk_pct}%**, placing you in the **{risk_level}** category. In addition, unsupervised clustering grouped your profile with the **{cluster_label}** (where the historical cohort prevalence was {cluster_prev}%).

### Primary Risk Drivers
The top clinical factors influencing this prediction include:
{drivers_md}

### Protective & Stabilizing Factors
Parameters that aligned with healthy normative ranges serve as positive clinical stabilizers. Keeping these within target boundaries helps buffer overall health risk.

### Actionable Next Steps & Questions for Your Doctor
- {advice}
- Bring this parameter summary to your next physician visit to discuss whether confirmatory lab tests (e.g. HbA1c, full lipid profile, or renal panel) are recommended.
- Explore gradual, sustainable adjustments to the modifiable lifestyle factors shown in the What-If analysis.

> *Source: Clinical Rule Engine (Offline Fallback — set GEMINI_API_KEY to activate Live LLM).*
"""


def explain_prediction(disease: str, input_dict: dict, prediction_result: dict) -> dict:
    """
    Generates a natural-language clinical explanation using Gemini LLM if available,
    otherwise returns a clinically tailored template.
    """
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    disease_title = prediction_result["disease_title"]
    risk_level = prediction_result["risk_level"]
    risk_pct = prediction_result["risk_percentage"]
    cluster_label = prediction_result["cluster_label"]
    top_features = prediction_result.get("top_features", [])

    if api_key:
        try:
            genai.configure(api_key=api_key)
            # Use gemini-1.5-flash or gemini-2.0-flash
            model = genai.GenerativeModel("gemini-1.5-flash")

            features_text = "\n".join([
                f"- {f['label']}: value {f['value']} ({f['direction']} relative to healthy controls, impact score {f['score']:.2f})"
                for f in top_features
            ])

            prompt = f"""You are a supportive, calm clinical health educator explaining a machine learning risk assessment for a college engineering project.
Do NOT give a medical diagnosis. Provide an objective, clear, empathetic plain-language explanation of the following patient results.

Disease Assessed: {disease_title}
Estimated Risk: {risk_pct}% ({risk_level})
Unsupervised Cluster: {cluster_label} (Cohort prevalence: {prediction_result['cluster_prevalence']}%)
Top Contributing Clinical Factors:
{features_text}

Format your response in Markdown with these exact section headings:
### Clinical Overview
(2-3 sentences explaining what {risk_pct}% risk means clearly without alarming the user)

### Primary Risk Drivers
(Bullet points discussing why the top 2-3 factors elevate risk from a physiological perspective)

### Protective & Stabilizing Factors
(Brief note highlighting any baseline strengths or stabilizing metrics)

### Actionable Next Steps & Questions for Your Doctor
(2-3 practical, evidence-based lifestyle suggestions and clinical questions to ask a doctor)

Keep the copy concise, professional, and accessible. Conclude with an educational disclaimer.
"""
            response = model.generate_content(prompt)
            if response and response.text:
                return {
                    "explanation": response.text.strip(),
                    "source": "Gemini 1.5 Flash (Generative AI)",
                    "status": "success",
                }
        except Exception as e:
            # Graceful fallback if API fails or quota exceeded
            fallback_text = generate_template_explanation(disease_title, prediction_result, input_dict)
            return {
                "explanation": fallback_text,
                "source": "Clinical Rule Engine (Fallback after API error)",
                "status": "fallback",
                "error_detail": str(e),
            }

    # If no API key provided
    fallback_text = generate_template_explanation(disease_title, prediction_result, input_dict)
    return {
        "explanation": fallback_text,
        "source": "Clinical Rule Engine (Offline Fallback)",
        "status": "fallback",
    }
