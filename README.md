# Autonomous Vehicle Safety Intelligence

An end-to-end machine learning application for screening serious/fatal crash severity patterns using the **NHTSA Crash Report Sampling System (CRSS) 2024** dataset.

The project combines crash-data preprocessing, supervised machine learning, threshold optimization, local model sensitivity analysis, what-if scenario comparison, and cloud deployment in an interactive Streamlit application.

## Live Demo

**Streamlit Application:**  
https://autonomous-vehicle-safety-engine.streamlit.app/

## Project Overview

Road crashes are influenced by complex interactions between roadway conditions, environmental factors, vehicle movement, collision configuration, and traffic context.

This project investigates whether machine learning can identify patterns associated with serious or fatal outcomes among reported crashes.

The system allows users to configure a crash scenario and obtain:

- Serious/fatal outcome score
- Risk-band interpretation
- Threshold-based classification
- Local model sensitivity analysis
- What-if scenario comparison
- Model performance information

> **Important:** This is a research demonstration. The model estimates patterns associated with serious/fatal outcomes among crashes represented in CRSS. It does not predict whether an autonomous vehicle will crash and should not be interpreted as a causal safety model.

---

## Dataset

**Source:** NHTSA Crash Report Sampling System (CRSS) 2024

Initial crash records:

- 51,658 crashes

After excluding records with unknown, unreported, or unsuitable severity outcomes:

- 50,654 usable crashes
- 44,069 Non-serious — 87.0%
- 6,585 Serious/Fatal — 13.0%

The target therefore presents a substantial class imbalance.

### Target Definition

The multiclass CRSS maximum injury-severity variable was transformed into a binary modeling target:

- **0 — Non-serious**
- **1 — Serious/Fatal**

The positive class combines suspected serious injury and fatal injury outcomes.

---

## Machine Learning Pipeline

The project progressed through several modeling stages:

1. Dummy majority-class baseline
2. Logistic Regression
3. Random Forest V1
4. XGBoost V1
5. Expanded 22-feature Random Forest V2
6. Expanded 22-feature XGBoost V2
7. Classification-threshold analysis
8. Production inference pipeline

Categorical variables are encoded as part of the preprocessing pipeline, while numerical inputs are handled using the trained preprocessing configuration.

---

## Model Comparison

| Model | Features | Accuracy | Serious Precision | Serious Recall | Serious F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Random Forest V1 | 13 | 0.6945 | 0.2382 | 0.6143 | 0.3433 | 0.7307 | 0.3023 |
| XGBoost V1 | 13 | 0.6527 | 0.2256 | 0.6872 | 0.3397 | 0.7291 | 0.3011 |
| Random Forest V2 | 22 | 0.7292 | 0.2671 | 0.6211 | 0.3736 | 0.7611 | 0.3355 |
| **XGBoost V2** | **22** | **0.6974** | **0.2554** | **0.6932** | **0.3733** | **0.7685** | **0.3582** |

XGBoost V2 was selected for the production application based primarily on its ranking performance and stronger PR-AUC in the imbalanced classification setting.

---

## Threshold Optimization

The default probability threshold of 0.50 was not automatically assumed to be optimal.

Several thresholds were evaluated:

| Threshold | Precision | Recall | F1 |
|---:|---:|---:|---:|
| 0.30 | 0.1864 | 0.9051 | 0.3092 |
| 0.40 | 0.2165 | 0.8322 | 0.3436 |
| 0.50 | 0.2554 | 0.6932 | 0.3733 |
| 0.55 | 0.2796 | 0.6128 | 0.3840 |
| **0.60** | **0.3135** | **0.5292** | **0.3938** |
| 0.65 | 0.3438 | 0.4328 | 0.3832 |
| 0.70 | 0.3852 | 0.3159 | 0.3471 |

A production threshold of **0.60** was selected because it produced the highest positive-class F1 score among the tested thresholds.

---

## Production Model

**Algorithm:** XGBoost V2  
**Features:** 22  
**Dataset:** NHTSA CRSS 2024  
**ROC-AUC:** 0.7685  
**PR-AUC:** 0.3582  
**Decision Threshold:** 0.60

At the selected threshold:

- Serious/Fatal Precision: 0.3135
- Serious/Fatal Recall: 0.5292
- Serious/Fatal F1: 0.3938

---

## Application Features

### Interactive Scenario Analysis

Users can configure conditions including:

- Region
- Urban/rural environment
- Month and day
- Time of day
- Lighting
- Weather
- Collision configuration
- Roadway relationship
- Intersection context
- Speed-related condition
- Speed limit
- Surface condition
- Road alignment
- Road profile
- Trafficway configuration
- Number of lanes
- Traffic control
- Pre-crash movement

The complete scenario is passed through the same preprocessing and XGBoost pipeline used by the production model.

### Local Sensitivity Analysis

For an analyzed scenario, the application performs local perturbation analysis by modifying individual inputs while holding the remaining scenario constant.

The interface reports how these perturbations change the model score.

This analysis describes **model sensitivity**, not causal effects, and should not be interpreted as SHAP values or causal feature importance.

### What-If Scenario Comparison

Users can modify selected environmental conditions such as:

- Lighting
- Weather
- Road surface

The application compares the original and modified model scores and reports the change in percentage points.

This provides an interactive demonstration of model behavior under alternative inputs.

---

## Technology Stack

- Python
- Pandas
- Scikit-learn
- XGBoost
- Joblib
- Streamlit
- Git
- GitHub
- Kaggle

---

## Project Structure

```text
autonomous-vehicle-safety-engine/
│
├── app.py
├── requirements.txt
├── README.md
│
├── models/
│   ├── av_safety_xgb_v2.joblib
│   ├── category_options.json
│   └── model_config.json
│
├── src/
│   └── inference.py
│
└── tests/
    └── test_inference.py
