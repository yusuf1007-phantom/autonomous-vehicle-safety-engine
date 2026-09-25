from pathlib import Path
import json
import joblib
import pandas as pd


# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_PATH = MODEL_DIR / "av_safety_xgb_v2.joblib"
CONFIG_PATH = MODEL_DIR / "model_config.json"
OPTIONS_PATH = MODEL_DIR / "category_options.json"


# ---------------------------------------------------------
# LOAD PRODUCTION ASSETS
# ---------------------------------------------------------

def _load_json(path):
    with open(path, "r") as f:
        return json.load(f)


MODEL = joblib.load(MODEL_PATH)
CONFIG = _load_json(CONFIG_PATH)
CATEGORY_OPTIONS = _load_json(OPTIONS_PATH)

THRESHOLD = float(CONFIG.get("threshold", 0.60))
FEATURES = CONFIG["features"]


# ---------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------

def validate_scenario(scenario):
    """
    Ensure that the incoming scenario contains every feature
    expected by the trained production model.
    """

    missing = [
        feature
        for feature in FEATURES
        if feature not in scenario
    ]

    if missing:
        raise ValueError(
            f"Missing required features: {missing}"
        )

    return True


# ---------------------------------------------------------
# RISK BAND
# ---------------------------------------------------------

def get_risk_band(score):
    """
    Portfolio/demo interpretation bands.

    These are interface bands, not validated autonomous-
    vehicle safety thresholds.
    """

    if score < 0.30:
        return "Lower"

    if score < THRESHOLD:
        return "Elevated"

    return "High"


# ---------------------------------------------------------
# MODEL INFERENCE
# ---------------------------------------------------------

def analyze_scenario(scenario):
    """
    Run one driving scenario through the complete trained
    preprocessing + XGBoost pipeline.
    """

    validate_scenario(scenario)

    ordered_scenario = {
        feature: scenario[feature]
        for feature in FEATURES
    }

    scenario_df = pd.DataFrame([ordered_scenario])

    score = float(
        MODEL.predict_proba(scenario_df)[0, 1]
    )

    flag = score >= THRESHOLD

    return {
        "score": score,
        "score_percent": score * 100,
        "threshold": THRESHOLD,
        "flag": bool(flag),
        "classification": (
            "Serious/Fatal Risk Flag"
            if flag
            else "No Serious/Fatal Risk Flag"
        ),
        "risk_band": get_risk_band(score),
    }


# ---------------------------------------------------------
# SCENARIO COMPARISON
# ---------------------------------------------------------

def compare_scenarios(original, modified):
    """
    Compare model sensitivity between two scenarios.

    The score difference must NOT be interpreted as a
    causal effect.
    """

    original_result = analyze_scenario(original)
    modified_result = analyze_scenario(modified)

    difference = (
        modified_result["score"]
        - original_result["score"]
    )

    return {
        "original": original_result,
        "modified": modified_result,
        "absolute_change": difference,
        "percentage_point_change": difference * 100,
    }

# ---------------------------------------------------------
# LOCAL SCENARIO EXPLANATION
# ---------------------------------------------------------

def explain_scenario(scenario, max_features=6):
    """
    Estimate local feature sensitivity for one scenario.

    Each feature is replaced with an alternative valid category
    (or a small numerical perturbation), while all other features
    remain unchanged.

    This is a model-sensitivity explanation, not SHAP and not
    a causal interpretation.
    """

    validate_scenario(scenario)

    baseline_result = analyze_scenario(scenario)
    baseline_score = baseline_result["score"]

    explanations = []

    for feature in FEATURES:

        modified = scenario.copy()
        original_value = scenario[feature]

        # ---------------------------------------------
        # CATEGORICAL FEATURES
        # ---------------------------------------------
        if feature in CATEGORY_OPTIONS:

            options = CATEGORY_OPTIONS[feature]

            alternatives = [
                value
                for value in options
                if value != original_value
            ]

            if not alternatives:
                continue

            # Use the first valid alternative as a
            # reference perturbation.
            reference_value = alternatives[0]

            modified[feature] = reference_value

        # ---------------------------------------------
        # NUMERICAL FEATURES
        # ---------------------------------------------
        elif isinstance(original_value, (int, float)):

            # Small interpretable perturbation.
            reference_value = original_value + 1
            modified[feature] = reference_value

        else:
            continue

        try:
            modified_result = analyze_scenario(modified)
        except Exception:
            continue

        modified_score = modified_result["score"]

        # Positive contribution means the current value
        # produces a higher model score than the reference.
        effect = baseline_score - modified_score

        explanations.append({
            "feature": feature,
            "current_value": original_value,
            "reference_value": reference_value,
            "effect": effect,
            "effect_percentage_points": effect * 100,
            "direction": (
                "Higher model score"
                if effect > 0
                else "Lower model score"
                if effect < 0
                else "No measurable change"
            )
        })

    explanations.sort(
        key=lambda item: abs(item["effect"]),
        reverse=True
    )

    return explanations[:max_features]
# ---------------------------------------------------------
# MODEL INFORMATION
# ---------------------------------------------------------

def get_model_info():
    return {
        "name": CONFIG.get("model_name", "XGBoost V2"),
        "version": CONFIG.get("version", "1.0"),
        "dataset": CONFIG.get("dataset", "NHTSA CRSS 2024"),
        "roc_auc": CONFIG.get("roc_auc"),
        "pr_auc": CONFIG.get("pr_auc"),
        "threshold": THRESHOLD,
        "features": FEATURES,
    }


def get_category_options():
    return CATEGORY_OPTIONS