import streamlit as st
import pandas as pd

FEATURE_LABELS = {
    "REGIONNAME": "Region",
    "URBANICITYNAME": "Area Type",
    "MONTHNAME": "Month",
    "DAY_WEEKNAME": "Day of Week",
    "HOUR": "Hour",
    "MAN_COLLNAME": "Collision Type",
    "RELJCT1NAME": "Junction Related",
    "RELJCT2NAME": "Junction Type",
    "TYP_INTNAME": "Intersection Type",
    "REL_ROADNAME": "Roadway Location",
    "LGT_CONDNAME": "Lighting Condition",
    "WEATHERNAME": "Weather",
    "INT_HWYNAME": "Interstate Highway",
    "SPEEDRELNAME": "Speed Related",
    "VSPD_LIMNAME": "Speed Limit",
    "VSURCONDNAME": "Road Surface",
    "VALIGNNAME": "Road Alignment",
    "VPROFILENAME": "Road Profile",
    "VTRAFWAYNAME": "Trafficway Type",
    "VNUM_LANNAME": "Number of Lanes",
    "VTRAFCONNAME": "Traffic Control",
    "P_CRASH1NAME": "Pre-Crash Movement",
}
from src.inference import (
    analyze_scenario,
    explain_scenario,
    get_model_info,
    get_category_options,
)

st.set_page_config(
    page_title="AV Safety Intelligence Engine",
    page_icon="🚘",
    layout="wide",
)

info = get_model_info()
options = get_category_options()

st.title("🚘 Autonomous Vehicle Safety Intelligence Engine")

st.caption(
    "ML-powered crash severity screening using NHTSA CRSS 2024 data"
)

st.info(
    "Research demonstration only. "
    "This model estimates patterns associated with serious/fatal outcomes "
    "among reported crashes; it does not predict whether a vehicle will crash."
)

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

with st.sidebar:
    st.header("Model Information")

    st.write(f"**Model:** {info['name']}")
    st.write(f"**Dataset:** {info['dataset']}")
    st.write(f"**Features:** {len(info['features'])}")
    st.write(f"**ROC-AUC:** {info['roc_auc']:.4f}")
    st.write(f"**PR-AUC:** {info['pr_auc']:.4f}")
    st.write(f"**Decision threshold:** {info['threshold']:.2f}")

# ---------------------------------------------------------
# INPUTS
# ---------------------------------------------------------

st.header("Driving Scenario")

st.write(
    "Configure a crash scenario below and let the trained model "
    "evaluate its serious/fatal outcome score."
)

col1, col2 = st.columns(2)

scenario = {}

with col1:

    st.subheader("Environment & Crash Context")

    scenario["REGIONNAME"] = st.selectbox(
        "Region",
        options["REGIONNAME"]
    )

    scenario["URBANICITYNAME"] = st.selectbox(
        "Area Type",
        options["URBANICITYNAME"]
    )

    scenario["MONTHNAME"] = st.selectbox(
        "Month",
        options["MONTHNAME"]
    )

    scenario["DAY_WEEKNAME"] = st.selectbox(
        "Day of Week",
        options["DAY_WEEKNAME"]
    )

    scenario["HOUR"] = st.slider(
        "Hour of Day",
        0,
        23,
        12
    )

    scenario["LGT_CONDNAME"] = st.selectbox(
        "Lighting Condition",
        options["LGT_CONDNAME"]
    )

    scenario["WEATHERNAME"] = st.selectbox(
        "Weather",
        options["WEATHERNAME"]
    )

    scenario["REL_ROADNAME"] = st.selectbox(
        "Roadway Relation",
        options["REL_ROADNAME"]
    )

    scenario["MAN_COLLNAME"] = st.selectbox(
        "Collision Configuration",
        options["MAN_COLLNAME"]
    )

    scenario["RELJCT1NAME"] = st.selectbox(
        "Junction Relation",
        options["RELJCT1NAME"]
    )

    scenario["RELJCT2NAME"] = st.selectbox(
        "Junction Type",
        options["RELJCT2NAME"]
    )

with col2:

    st.subheader("Vehicle & Road Context")

    scenario["TYP_INTNAME"] = st.selectbox(
        "Intersection Type",
        options["TYP_INTNAME"]
    )

    scenario["INT_HWYNAME"] = st.selectbox(
        "Interstate Highway",
        options["INT_HWYNAME"]
    )

    scenario["SPEEDRELNAME"] = st.selectbox(
        "Speed Related",
        options["SPEEDRELNAME"]
    )

    scenario["VSPD_LIMNAME"] = st.selectbox(
        "Speed Limit",
        options["VSPD_LIMNAME"]
    )

    scenario["VSURCONDNAME"] = st.selectbox(
        "Road Surface",
        options["VSURCONDNAME"]
    )

    scenario["VALIGNNAME"] = st.selectbox(
        "Road Alignment",
        options["VALIGNNAME"]
    )

    scenario["VPROFILENAME"] = st.selectbox(
        "Road Profile",
        options["VPROFILENAME"]
    )

    scenario["VTRAFWAYNAME"] = st.selectbox(
        "Trafficway",
        options["VTRAFWAYNAME"]
    )

    scenario["VNUM_LANNAME"] = st.selectbox(
        "Number of Lanes",
        options["VNUM_LANNAME"]
    )

    scenario["VTRAFCONNAME"] = st.selectbox(
        "Traffic Control",
        options["VTRAFCONNAME"]
    )

    scenario["P_CRASH1NAME"] = st.selectbox(
        "Pre-Crash Movement",
        options["P_CRASH1NAME"]
    )

# ---------------------------------------------------------
# ANALYSIS
# ---------------------------------------------------------

st.divider()

if st.button(
    "Analyze Scenario",
    type="primary",
    use_container_width=True
):

    result = analyze_scenario(scenario)

    # Store the latest prediction and scenario so that we can
    # compare it with modified driving conditions later.
    st.session_state["baseline_result"] = result
    st.session_state["baseline_scenario"] = scenario.copy()

    st.header("Safety Intelligence")

    metric1, metric2, metric3 = st.columns(3)

    metric1.metric(
        "Serious/Fatal Score",
        f"{result['score_percent']:.1f}%"
    )

    metric2.metric(
        "Risk Band",
        result["risk_band"]
    )

    metric3.metric(
        "Decision Threshold",
        f"{result['threshold'] * 100:.0f}%"
    )

    st.progress(
        min(max(result["score"], 0.0), 1.0)
    )

    if result["flag"]:
        st.warning(
            "⚠️ The model score exceeds the selected screening threshold."
        )
    else:
        st.success(
            "✓ The model score is below the selected screening threshold."
        )

    st.subheader("Model Classification")

    st.write(f"**{result['classification']}**")

    st.caption(
        "The score reflects statistical patterns learned from "
        "reported crashes in NHTSA CRSS 2024. It should not be "
        "interpreted as a causal safety assessment or the probability "
        "that an autonomous vehicle will crash."
    )
    # -----------------------------------------------------
    # LOCAL MODEL EXPLANATION
    # -----------------------------------------------------

    st.subheader("Prediction Explanation")

    st.caption(
        "The values below show how sensitive this prediction is "
        "to individual input changes. They are model-behavior "
        "indicators, not causal effects."
    )

    explanations = explain_scenario(
        scenario,
        max_features=6
    )

    if explanations:

        explanation_df = pd.DataFrame([
            {
                "Feature": item["feature"],
                "Current Value": item["current_value"],
                "Compared With": item["reference_value"],
                "Score Effect": (
                    f"{item['effect_percentage_points']:+.1f} pp"
                ),
                "Direction": item["direction"],
            }
            for item in explanations
        ])
        explanation_df["Feature"] = explanation_df["Feature"].replace(
            FEATURE_LABELS
        )


        st.dataframe(
            explanation_df,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "No local sensitivity explanation was available "
            "for this scenario."
        )    
# ---------------------------------------------------------
# WHAT-IF SCENARIO COMPARISON
# ---------------------------------------------------------

if "baseline_result" in st.session_state:

    st.divider()
    st.header("What-If Scenario Analysis")

    st.write(
        "Modify selected driving conditions and compare the model's "
        "serious/fatal outcome score with the original scenario."
    )

    baseline_result = st.session_state["baseline_result"]
    baseline_scenario = st.session_state["baseline_scenario"]

    comparison_scenario = baseline_scenario.copy()

    col1, col2, col3 = st.columns(3)

    with col1:
        comparison_scenario["LGT_CONDNAME"] = st.selectbox(
            "Modified Lighting",
            [
                "Daylight",
                "Dark - Lighted",
                "Dark - Not Lighted",
                "Dawn",
                "Dusk"
            ],
            index=(
                [
                    "Daylight",
                    "Dark - Lighted",
                    "Dark - Not Lighted",
                    "Dawn",
                    "Dusk"
                ].index(baseline_scenario["LGT_CONDNAME"])
                if baseline_scenario["LGT_CONDNAME"] in [
                    "Daylight",
                    "Dark - Lighted",
                    "Dark - Not Lighted",
                    "Dawn",
                    "Dusk"
                ]
                else 0
            ),
            key="whatif_light"
        )

    with col2:
        comparison_scenario["WEATHERNAME"] = st.selectbox(
            "Modified Weather",
            [
                "Clear",
                "Rain",
                "Snow",
                "Fog, Smog, Smoke",
                "Severe Crosswinds"
            ],
            index=(
                [
                    "Clear",
                    "Rain",
                    "Snow",
                    "Fog, Smog, Smoke",
                    "Severe Crosswinds"
                ].index(baseline_scenario["WEATHERNAME"])
                if baseline_scenario["WEATHERNAME"] in [
                    "Clear",
                    "Rain",
                    "Snow",
                    "Fog, Smog, Smoke",
                    "Severe Crosswinds"
                ]
                else 0
            ),
            key="whatif_weather"
        )

    with col3:
        comparison_scenario["VSURCONDNAME"] = st.selectbox(
            "Modified Road Surface",
            [
                "Dry",
                "Wet",
                "Snow",
                "Ice/Frost",
                "Slush"
            ],
            index=(
                [
                    "Dry",
                    "Wet",
                    "Snow",
                    "Ice/Frost",
                    "Slush"
                ].index(baseline_scenario["VSURCONDNAME"])
                if baseline_scenario["VSURCONDNAME"] in [
                    "Dry",
                    "Wet",
                    "Snow",
                    "Ice/Frost",
                    "Slush"
                ]
                else 0
            ),
            key="whatif_surface"
        )

    if st.button(
        "Compare Scenario",
        use_container_width=True
    ):

        comparison_result = analyze_scenario(comparison_scenario)

        original_score = baseline_result["score_percent"]
        modified_score = comparison_result["score_percent"]
        difference = modified_score - original_score

        st.subheader("Scenario Comparison")

        original_col, arrow_col, modified_col = st.columns(
            [2, 1, 2]
        )

        with original_col:
            st.metric(
                "Original Scenario",
                f"{original_score:.1f}%"
            )
            st.caption(
                f"Risk band: {baseline_result['risk_band']}"
            )

        with arrow_col:
            st.markdown("### →")

        with modified_col:
            st.metric(
                "Modified Scenario",
                f"{modified_score:.1f}%",
                delta=f"{difference:+.1f} percentage points"
            )
            st.caption(
                f"Risk band: {comparison_result['risk_band']}"
            )

        if difference > 0:
            st.warning(
                f"The modified scenario increased the model score "
                f"by {abs(difference):.1f} percentage points."
            )

        elif difference < 0:
            st.success(
                f"The modified scenario decreased the model score "
                f"by {abs(difference):.1f} percentage points."
            )

        else:
            st.info(
                "The modified conditions did not change the model score."
            )

        st.caption(
                "Local sensitivity analysis. Each row compares the current input "
                "with an alternative valid value while keeping the remaining scenario "
                "unchanged. Score effects describe model behavior and should not be "
                "interpreted as causal effects."
            )
# ---------------------------------------------------------
# MODEL PERFORMANCE
# ---------------------------------------------------------

st.divider()

with st.expander("Model performance & methodology"):

    performance = pd.DataFrame({
        "Metric": [
            "ROC-AUC",
            "PR-AUC",
            "Production Threshold",
            "Threshold Precision",
            "Threshold Recall",
            "Threshold F1"
        ],
        "Value": [
            info["roc_auc"],
            info["pr_auc"],
            info["threshold"],
            0.3135,
            0.5292,
            0.3938
        ]
    })

    st.dataframe(
        performance,
        use_container_width=True,
        hide_index=True
    )

    st.write(
        "The production threshold of 0.60 was selected because it "
        "produced the highest F1 score among the tested thresholds "
        "from 0.30 through 0.70."
    )

st.caption(
    "AV Safety Intelligence Engine •  Research Project"
)