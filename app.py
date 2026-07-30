from __future__ import annotations

# import libraries
import json
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# configure the streamlit page
st.set_page_config(
    page_title="Intent / Shopper Purchase Predictor",
    page_icon="◼",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# define file paths and accepted model filenames
app_dir = Path(__file__).resolve().parent
model_candidates = [
    app_dir / "online_shopper_model.joblib",
    app_dir / "model.pkl",
    app_dir / "Online_shopper_intention_GB_model",
    app_dir / "Online_shopper_intention_GB_model.joblib",
]


# define the available input choices
month_options = [
    "Feb",
    "Mar",
    "May",
    "June",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]

visitor_options = [
    "Returning_Visitor",
    "New_Visitor",
    "Other",
]


# define the starting values shown in the form
default_inputs = {
    "administrative": 2,
    "administrative_duration_min": 2.0,
    "informational": 0,
    "informational_duration_min": 0.0,
    "product_related": 18,
    "product_related_duration_min": 14.0,
    "bounce_rate": 0.020,
    "exit_rate": 0.045,
    "page_values": 4.0,
    "special_day": 0.0,
    "month": "Nov",
    "weekend": False,
    "operating_system": 2,
    "browser": 2,
    "region": 1,
    "traffic_type": 2,
    "visitor_type": "Returning_Visitor",
}


# define ready-made examples for demonstrations
presets = {
    "High intent": {
        "administrative": 5,
        "administrative_duration_min": 5.0,
        "informational": 1,
        "informational_duration_min": 1.5,
        "product_related": 55,
        "product_related_duration_min": 42.0,
        "bounce_rate": 0.003,
        "exit_rate": 0.018,
        "page_values": 42.0,
        "special_day": 0.2,
        "month": "Nov",
        "weekend": True,
        "operating_system": 2,
        "browser": 2,
        "region": 1,
        "traffic_type": 2,
        "visitor_type": "Returning_Visitor",
    },
    "Active browser": {
        "administrative": 1,
        "administrative_duration_min": 0.5,
        "informational": 0,
        "informational_duration_min": 0.0,
        "product_related": 34,
        "product_related_duration_min": 28.0,
        "bounce_rate": 0.018,
        "exit_rate": 0.052,
        "page_values": 5.0,
        "special_day": 0.0,
        "month": "May",
        "weekend": False,
        "operating_system": 2,
        "browser": 2,
        "region": 3,
        "traffic_type": 3,
        "visitor_type": "Returning_Visitor",
    },
    "Low intent": {
        "administrative": 0,
        "administrative_duration_min": 0.0,
        "informational": 0,
        "informational_duration_min": 0.0,
        "product_related": 4,
        "product_related_duration_min": 2.0,
        "bounce_rate": 0.150,
        "exit_rate": 0.180,
        "page_values": 0.0,
        "special_day": 0.0,
        "month": "Mar",
        "weekend": False,
        "operating_system": 3,
        "browser": 2,
        "region": 1,
        "traffic_type": 1,
        "visitor_type": "New_Visitor",
    },
}


# apply the clean editorial design
st.markdown(
    """
    <style>
        :root {
            --paper: #f3f1eb;
            --surface: #fbfaf7;
            --ink: #111111;
            --muted: #6f6d67;
            --line: #d6d2c8;
            --soft: #e9e5dc;
            --accent: #c85b3c;
        }

        html, body, [class*="css"] {
            font-family: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont,
                "Segoe UI", Helvetica, Arial, sans-serif;
        }

        .stApp {
            background: var(--paper);
            color: var(--ink);
        }

        .block-container {
            max-width: 1320px;
            padding: 1.15rem 2.25rem 4.5rem;
        }

        header[data-testid="stHeader"] {
            background: transparent;
        }

        #MainMenu, footer {
            visibility: hidden;
        }

        [data-testid="stSidebar"], [data-testid="collapsedControl"] {
            display: none;
        }

        .top-nav {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1.5rem;
            padding: 0.35rem 0 1rem;
            border-bottom: 1px solid var(--ink);
            margin-bottom: 1.25rem;
        }

        .brand {
            font-size: 1rem;
            font-weight: 850;
            letter-spacing: -0.055em;
        }

        .nav-items {
            display: flex;
            gap: 2rem;
            color: var(--ink);
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
        }

        .hero-grid {
            display: grid;
            grid-template-columns: minmax(0, 1.25fr) minmax(310px, 0.75fr);
            min-height: 430px;
            border-bottom: 1px solid var(--ink);
            margin-bottom: 2.5rem;
        }

        .hero-copy {
            display: flex;
            flex-direction: column;
            justify-content: flex-end;
            padding: 2rem 3rem 2.25rem 0;
        }

        .kicker {
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0.13em;
            text-transform: uppercase;
            margin-bottom: 1rem;
        }

        .hero-copy h1 {
            color: var(--ink);
            font-size: clamp(3.45rem, 7vw, 6.8rem);
            line-height: 0.89;
            letter-spacing: -0.078em;
            font-weight: 500;
            margin: 0;
            max-width: 900px;
        }

        .hero-copy p {
            color: var(--muted);
            max-width: 680px;
            font-size: 1rem;
            line-height: 1.7;
            margin: 1.55rem 0 0;
        }

        .hero-panel {
            position: relative;
            overflow: hidden;
            background: var(--ink);
            color: var(--surface);
            padding: 2rem;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }

        .hero-panel:after {
            content: "";
            position: absolute;
            width: 260px;
            height: 260px;
            border: 1px solid rgba(255,255,255,0.23);
            border-radius: 50%;
            right: -88px;
            bottom: -104px;
        }

        .panel-index {
            font-size: 0.72rem;
            letter-spacing: 0.15em;
            text-transform: uppercase;
            color: rgba(255,255,255,0.66);
        }

        .panel-statement {
            position: relative;
            z-index: 1;
            font-size: clamp(1.55rem, 3vw, 2.75rem);
            line-height: 1.04;
            letter-spacing: -0.05em;
            max-width: 390px;
            font-weight: 500;
        }

        .panel-meta {
            position: relative;
            z-index: 1;
            border-top: 1px solid rgba(255,255,255,0.35);
            padding-top: 1rem;
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
            color: rgba(255,255,255,0.72);
            font-size: 0.78rem;
            line-height: 1.5;
        }

        .section-head {
            display: grid;
            grid-template-columns: 105px 1fr;
            gap: 1.25rem;
            border-top: 1px solid var(--ink);
            padding-top: 1rem;
            margin: 2.5rem 0 1rem;
        }

        .section-index {
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0.13em;
            text-transform: uppercase;
        }

        .section-head h2 {
            color: var(--ink);
            font-size: clamp(1.75rem, 3vw, 3rem);
            line-height: 1;
            letter-spacing: -0.055em;
            font-weight: 500;
            margin: 0;
        }

        .section-head p {
            color: var(--muted);
            margin: 0.75rem 0 0;
            max-width: 750px;
            line-height: 1.65;
        }

        div[data-testid="stForm"] {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 0;
            padding: 1.25rem 1.35rem 1.35rem;
            box-shadow: none;
        }

        div[data-testid="stForm"] h3,
        div[data-testid="stForm"] h4 {
            color: var(--ink);
            letter-spacing: -0.035em;
        }

        div[data-testid="stMetric"] {
            background: transparent;
            border-top: 1px solid var(--ink);
            border-radius: 0;
            padding: 0.75rem 0;
        }

        div[data-testid="stMetric"] label {
            color: var(--muted);
            font-size: 0.72rem;
            letter-spacing: 0.06em;
            text-transform: uppercase;
        }

        div[data-testid="stMetricValue"] {
            color: var(--ink);
            letter-spacing: -0.045em;
        }

        .stButton > button,
        .stDownloadButton > button {
            width: 100%;
            min-height: 2.8rem;
            border-radius: 0;
            border: 1px solid var(--ink);
            background: transparent;
            color: var(--ink);
            font-weight: 750;
            letter-spacing: 0.01em;
            transition: all 0.18s ease;
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover {
            background: var(--ink);
            color: var(--surface);
            border-color: var(--ink);
        }

        .stFormSubmitButton > button {
            width: 100%;
            min-height: 3.1rem;
            border-radius: 0;
            border: 1px solid var(--ink);
            background: var(--ink);
            color: white;
            font-weight: 800;
            letter-spacing: 0.02em;
        }

        .stFormSubmitButton > button:hover {
            background: var(--accent);
            border-color: var(--accent);
            color: white;
        }

        div[data-baseweb="input"] > div,
        div[data-baseweb="select"] > div,
        div[data-testid="stNumberInputContainer"] > div,
        .stTextInput input {
            border-radius: 0 !important;
        }

        div[data-baseweb="slider"] {
            padding-top: 0.2rem;
        }

        .active-note {
            color: var(--muted);
            font-size: 0.82rem;
            margin-top: 0.5rem;
        }

        .empty-state {
            min-height: 260px;
            display: grid;
            place-items: center;
            text-align: center;
            border: 1px solid var(--line);
            background: var(--surface);
            padding: 2rem;
        }

        .empty-state h3 {
            color: var(--ink);
            font-size: 1.8rem;
            font-weight: 500;
            letter-spacing: -0.045em;
            margin: 0 0 0.65rem;
        }

        .empty-state p {
            color: var(--muted);
            max-width: 560px;
            line-height: 1.65;
            margin: 0;
        }

        .result-grid {
            display: grid;
            grid-template-columns: 0.82fr 1.18fr;
            border: 1px solid var(--ink);
            background: var(--surface);
            min-height: 340px;
        }

        .probability-panel {
            padding: 2rem;
            border-right: 1px solid var(--ink);
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }

        .probability-number {
            color: var(--ink);
            font-size: clamp(4.5rem, 10vw, 8rem);
            line-height: 0.9;
            letter-spacing: -0.085em;
            font-weight: 450;
        }

        .probability-label {
            color: var(--muted);
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0.12em;
            text-transform: uppercase;
        }

        .probability-track {
            height: 8px;
            background: var(--soft);
            margin-top: 1rem;
            overflow: hidden;
        }

        .probability-fill {
            height: 100%;
            background: var(--accent);
        }

        .decision-panel {
            background: var(--ink);
            color: var(--surface);
            padding: 2rem;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }

        .decision-panel h2 {
            color: var(--surface);
            font-size: clamp(2.25rem, 5vw, 4.65rem);
            line-height: 0.95;
            letter-spacing: -0.07em;
            font-weight: 500;
            margin: 0.6rem 0 1rem;
        }

        .decision-panel p {
            color: rgba(255,255,255,0.72);
            max-width: 620px;
            line-height: 1.65;
            margin: 0;
        }

        .decision-footer {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
            border-top: 1px solid rgba(255,255,255,0.3);
            padding-top: 1rem;
            margin-top: 2rem;
            font-size: 0.8rem;
            color: rgba(255,255,255,0.72);
        }

        .signal-row {
            display: grid;
            grid-template-columns: 170px 1fr;
            gap: 1rem;
            padding: 1rem 0;
            border-bottom: 1px solid var(--line);
        }

        .signal-row strong {
            color: var(--ink);
            font-size: 0.85rem;
        }

        .signal-row span {
            color: var(--muted);
            line-height: 1.55;
        }

        .notice {
            border-left: 3px solid var(--accent);
            background: var(--surface);
            padding: 1rem 1.1rem;
            color: var(--ink);
            line-height: 1.6;
        }

        .fine-print {
            color: var(--muted);
            font-size: 0.8rem;
            line-height: 1.6;
        }

        .footer-note {
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            border-top: 1px solid var(--ink);
            padding-top: 1rem;
            margin-top: 3rem;
            color: var(--muted);
            font-size: 0.75rem;
            letter-spacing: 0.03em;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 1.6rem;
            border-bottom: 1px solid var(--line);
        }

        .stTabs [data-baseweb="tab"] {
            height: 3rem;
            padding: 0;
            color: var(--muted);
            font-weight: 700;
        }

        .stTabs [aria-selected="true"] {
            color: var(--ink) !important;
        }

        @media (max-width: 900px) {
            .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
            }

            .nav-items {
                display: none;
            }

            .hero-grid,
            .result-grid {
                grid-template-columns: 1fr;
            }

            .hero-copy {
                min-height: 390px;
                padding-right: 0;
            }

            .probability-panel {
                border-right: 0;
                border-bottom: 1px solid var(--ink);
            }

            .section-head {
                grid-template-columns: 1fr;
                gap: 0.5rem;
            }

            .signal-row {
                grid-template-columns: 1fr;
                gap: 0.35rem;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# find the saved model file in the project folder
def find_model_path() -> Path | None:
    for candidate in model_candidates:
        if candidate.exists():
            return candidate
    return None


# load and validate the saved model bundle
@st.cache_resource
def load_model_bundle(path: Path) -> dict[str, Any]:
    loaded_object = joblib.load(path)

    if isinstance(loaded_object, dict):
        if "model" not in loaded_object:
            raise KeyError("The saved bundle does not contain a 'model' item.")
        if "feature_columns" not in loaded_object:
            raise KeyError("The saved bundle does not contain a 'feature_columns' item.")

        bundle = dict(loaded_object)
        bundle["feature_columns"] = list(bundle["feature_columns"])
        return bundle

    if hasattr(loaded_object, "feature_names_in_"):
        return {
            "model": loaded_object,
            "feature_columns": list(loaded_object.feature_names_in_),
            "threshold": 0.50,
            "model_name": loaded_object.__class__.__name__,
        }

    raise TypeError(
        "The saved file must be a model bundle containing 'model' and "
        "'feature_columns'."
    )


# create the session variables used by the interactive form
def initialise_session_state() -> None:
    for key, value in default_inputs.items():
        st.session_state.setdefault(key, value)

    st.session_state.setdefault("active_preset", "Custom session")
    st.session_state.setdefault("last_prediction", None)
    st.session_state.setdefault("prediction_history", [])


# load a ready-made session into the form
def apply_preset(preset_name: str) -> None:
    for key, value in presets[preset_name].items():
        st.session_state[key] = value

    st.session_state["active_preset"] = preset_name
    st.session_state["last_prediction"] = None
    st.rerun()


# reset the form back to its original values
def reset_form() -> None:
    for key, value in default_inputs.items():
        st.session_state[key] = value

    st.session_state["active_preset"] = "Custom session"
    st.session_state["last_prediction"] = None
    st.rerun()


# place a value into the model input only when that feature exists
def set_if_present(
    frame: pd.DataFrame,
    feature_name: str,
    value: float | int,
) -> None:
    if feature_name in frame.columns:
        frame.at[0, feature_name] = value


# activate the matching one-hot encoded category column
def set_category(
    frame: pd.DataFrame,
    base_name: str,
    value: str | int | bool,
) -> None:
    columns = set(frame.columns)

    if base_name in columns:
        if isinstance(value, bool):
            frame.at[0, base_name] = int(value)
        elif isinstance(value, (int, float)):
            frame.at[0, base_name] = value

    candidates = [
        f"{base_name}_{value}",
        f"{base_name}_{str(value)}",
    ]

    if isinstance(value, bool):
        candidates.extend(
            [
                f"{base_name}_{int(value)}",
                f"{base_name}_{str(value).lower()}",
            ]
        )

    for candidate in candidates:
        if candidate in columns:
            frame.at[0, candidate] = 1.0


# convert the visible form values into the exact model feature layout
def build_model_input(
    inputs: dict[str, Any],
    feature_columns: list[str],
) -> pd.DataFrame:
    row = pd.DataFrame(
        np.zeros((1, len(feature_columns))),
        columns=feature_columns,
    )

    administrative_duration = inputs["administrative_duration_min"] * 60.0
    informational_duration = inputs["informational_duration_min"] * 60.0
    product_duration = inputs["product_related_duration_min"] * 60.0

    numerical_values = {
        "Administrative": inputs["administrative"],
        "Administrative_Duration": administrative_duration,
        "Informational": inputs["informational"],
        "Informational_Duration": informational_duration,
        "ProductRelated": inputs["product_related"],
        "ProductRelated_Duration": product_duration,
        "BounceRates": inputs["bounce_rate"],
        "ExitRates": inputs["exit_rate"],
        "PageValues": inputs["page_values"],
        "SpecialDay": inputs["special_day"],
    }

    for feature_name, value in numerical_values.items():
        set_if_present(row, feature_name, value)

    engineered_values = {
        "AvgAdministrativeTime": (
            administrative_duration / inputs["administrative"]
            if inputs["administrative"]
            else 0.0
        ),
        "AvgInformationalTime": (
            informational_duration / inputs["informational"]
            if inputs["informational"]
            else 0.0
        ),
        "AvgProductTime": (
            product_duration / inputs["product_related"]
            if inputs["product_related"]
            else 0.0
        ),
        "TotalPages": (
            inputs["administrative"]
            + inputs["informational"]
            + inputs["product_related"]
        ),
        "TotalDuration": (
            administrative_duration
            + informational_duration
            + product_duration
        ),
        "VisitedAdministrative": int(inputs["administrative"] > 0),
        "VisitedInformational": int(inputs["informational"] > 0),
        "VisitedProductPage": int(inputs["product_related"] > 0),
        "HasPageValue": int(inputs["page_values"] > 0),
    }

    for feature_name, value in engineered_values.items():
        set_if_present(row, feature_name, value)

    set_category(row, "Month", inputs["month"])
    set_category(row, "Weekend", inputs["weekend"])
    set_category(row, "OperatingSystems", inputs["operating_system"])
    set_category(row, "Browser", inputs["browser"])
    set_category(row, "Region", inputs["region"])
    set_category(row, "TrafficType", inputs["traffic_type"])
    set_category(row, "VisitorType", inputs["visitor_type"])

    return row


# check the user inputs before running the model
def validate_inputs(inputs: dict[str, Any]) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    page_time_pairs = [
        (
            "administrative",
            "administrative_duration_min",
            "Administrative",
        ),
        (
            "informational",
            "informational_duration_min",
            "Informational",
        ),
        (
            "product_related",
            "product_related_duration_min",
            "Product-related",
        ),
    ]

    for count_key, duration_key, label in page_time_pairs:
        if inputs[count_key] == 0 and inputs[duration_key] > 0:
            errors.append(
                f"{label} time must be 0 when the {label.lower()} page count is 0."
            )

    total_pages = (
        inputs["administrative"]
        + inputs["informational"]
        + inputs["product_related"]
    )

    if total_pages == 0:
        errors.append(
            "Enter at least one visited page before analysing the session."
        )

    if inputs["page_values"] > 0 and inputs["product_related"] == 0:
        warnings.append(
            "Page value is above 0 while product-related pages are 0. "
            "Check that these values were entered correctly."
        )

    if inputs["bounce_rate"] >= 0.15:
        warnings.append(
            "The bounce rate is very high. The prediction can still run, "
            "but verify that the value is correct."
        )

    return errors, warnings


# calculate the probability of the purchase class
def predict_probability(model: Any, model_input: pd.DataFrame) -> float:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(model_input)[0]
        classes = list(getattr(model, "classes_", []))

        if 1 in classes:
            positive_index = classes.index(1)
        elif True in classes:
            positive_index = classes.index(True)
        else:
            positive_index = 1

        return float(probabilities[positive_index])

    if hasattr(model, "decision_function"):
        score = float(model.decision_function(model_input)[0])
        return float(1 / (1 + np.exp(-score)))

    raise TypeError(
        "The saved model does not support probability predictions."
    )


# translate the probability into a clear customer stage
def intent_segment(
    probability: float,
    threshold: float,
) -> tuple[str, str]:
    if probability >= 0.75:
        return (
            "High purchase intent",
            "Prioritise relevant recommendations and remove checkout friction.",
        )

    if probability >= threshold:
        return (
            "Likely purchaser",
            "Use a light conversion prompt, product reassurance, or cart reminder.",
        )

    if probability >= 0.25:
        return (
            "Consideration stage",
            "Support product discovery with reviews and useful information.",
        )

    return (
        "Low immediate intent",
        "Avoid costly promotions and focus on helpful future re-engagement.",
    )


# prepare simple explanations based on the submitted session
def build_business_signals(
    inputs: dict[str, Any],
) -> list[tuple[str, str]]:
    signals: list[tuple[str, str]] = []

    if inputs["page_values"] >= 20:
        signals.append(
            (
                "Strong page value",
                "This session contains a strong purchase-linked page signal.",
            )
        )
    elif inputs["page_values"] == 0:
        signals.append(
            (
                "No page value yet",
                "The session has not produced a purchase-linked page value.",
            )
        )
    else:
        signals.append(
            (
                "Developing page value",
                "The session has a positive but moderate page value.",
            )
        )

    if inputs["exit_rate"] <= 0.04:
        signals.append(
            (
                "Low exit risk",
                "The exit rate suggests that the visitor is continuing through the site.",
            )
        )
    elif inputs["exit_rate"] >= 0.12:
        signals.append(
            (
                "High exit risk",
                "The visitor may leave before completing a purchase.",
            )
        )

    if inputs["product_related"] >= 40:
        signals.append(
            (
                "Deep product exploration",
                "The visitor viewed a large number of product-related pages.",
            )
        )
    elif inputs["product_related"] <= 5:
        signals.append(
            (
                "Limited product engagement",
                "The visitor viewed only a small number of product-related pages.",
            )
        )

    if inputs["visitor_type"] == "Returning_Visitor":
        signals.append(
            (
                "Returning visitor",
                "The shopper has returned to the website and may be further along in the journey.",
            )
        )

    return signals[:4]


# create a clean global feature importance chart
def make_feature_importance_chart(
    model: Any,
    feature_columns: list[str],
) -> go.Figure | None:
    if hasattr(model, "feature_importances_"):
        importance_values = np.asarray(
            model.feature_importances_,
            dtype=float,
        )
    elif hasattr(model, "coef_"):
        importance_values = np.abs(np.asarray(model.coef_)[0])
    else:
        return None

    if len(importance_values) != len(feature_columns):
        return None

    importance_table = pd.DataFrame(
        {
            "Feature": feature_columns,
            "Importance": importance_values,
        }
    )

    importance_table = (
        importance_table.sort_values("Importance", ascending=False)
        .head(12)
        .sort_values("Importance")
    )

    figure = go.Figure(
        go.Bar(
            x=importance_table["Importance"],
            y=importance_table["Feature"],
            orientation="h",
            marker_color="#111111",
            hovertemplate=(
                "%{y}<br>importance: %{x:.4f}<extra></extra>"
            ),
        )
    )

    figure.update_layout(
        height=430,
        margin=dict(l=10, r=15, t=20, b=30),
        xaxis_title="global feature importance",
        yaxis_title="",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#111111"),
        xaxis=dict(showgrid=True, gridcolor="#d6d2c8"),
        yaxis=dict(showgrid=False),
    )

    return figure


# create the session state before drawing the widgets
initialise_session_state()


# draw the top navigation and hero section
st.markdown(
    """
    <div class="top-nav">
        <div class="brand">INTENT / COMMERCE</div>
    </div>

    <div class="hero-grid">
        <div class="hero-copy">
            <div class="kicker">purchase intention / session analysis</div>
            <h1>Understand intent before the session ends.</h1>
            <p>
                Enter a shopper's live session behaviour to estimate purchase
                probability, identify the current intent stage, and choose a
                suitable next action.
            </p>
        </div>
        <div class="hero-panel">
            <div class="panel-index">project / 01</div>
            <div class="panel-statement">
                A focused decision tool for e-commerce and digital marketing teams.
            </div>
            <div class="panel-meta">
                <div>Output<br><strong>Purchase probability</strong></div>
                <div>Use case<br><strong>Session prioritisation</strong></div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# locate and load the saved model
model_path = find_model_path()

if model_path is None:
    st.error(
        "The trained model file was not found. Place your saved model in the "
        "same folder as this Streamlit file."
    )
    st.code(
        "MLDP_project/\n"
        "├── streamlit_app.py\n"
        "├── online_shopper_model.joblib\n"
        "└── requirements.txt"
    )
    st.caption(
        "Accepted filenames: online_shopper_model.joblib, model.pkl, "
        "Online_shopper_intention_GB_model, or "
        "Online_shopper_intention_GB_model.joblib."
    )
    st.stop()

try:
    bundle = load_model_bundle(model_path)
except Exception as error:
    st.error(
        "The trained model could not be loaded. Export the model bundle again "
        "and make sure it contains the model and feature columns."
    )
    with st.expander("Show technical details"):
        st.exception(error)
    st.stop()

model = bundle["model"]
feature_columns = bundle["feature_columns"]
saved_threshold = float(bundle.get("threshold", 0.50))
model_name = str(bundle.get("model_name", model.__class__.__name__))
evaluation_metrics = bundle.get("evaluation_metrics", {})

if hasattr(model, "feature_names_in_"):
    if list(model.feature_names_in_) != feature_columns:
        st.error(
            "The saved feature order does not match the trained model. "
            "Export the model bundle again using the exact training columns."
        )
        st.stop()


# show the model controls and model status
st.markdown(
    """
    <div class="section-head">
        <div class="section-index">01 / setup</div>
        <div>
            <h2>Choose a session</h2>
            <p>
                Load a preset for a quick demonstration or enter a custom
                shopping session below.
            </p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

control_left, control_middle, control_right = st.columns([1.25, 1.25, 1])

with control_left:
    st.caption("model status")
    st.success(f"Loaded: {model_name}")

with control_middle:
    threshold = st.slider(
        "Purchase decision threshold",
        min_value=0.10,
        max_value=0.90,
        value=float(np.clip(saved_threshold, 0.10, 0.90)),
        step=0.01,
        help=(
            "Lowering the threshold usually identifies more purchasers, "
            "but can also increase false positives."
        ),
    )

with control_right:
    show_technical = st.checkbox(
        "Show technical details",
        value=False,
    )
    if st.button("Reset form", use_container_width=True):
        reset_form()

preset_columns = st.columns(3)

for column, preset_name in zip(preset_columns, presets):
    with column:
        if st.button(
            preset_name,
            key=f"preset_{preset_name}",
            use_container_width=True,
        ):
            apply_preset(preset_name)

st.markdown(
    f'<div class="active-note">active session: '
    f'<strong>{escape(st.session_state["active_preset"])}</strong></div>',
    unsafe_allow_html=True,
)


# collect the user input in one structured form
with st.form("shopper_session_form", clear_on_submit=False):
    st.markdown("### Session information")

    engagement_column, behaviour_column, context_column = st.columns(3)

    with engagement_column:
        st.markdown("#### Page engagement")

        st.number_input(
            "Administrative pages",
            min_value=0,
            max_value=30,
            step=1,
            key="administrative",
            help="Examples include account, login, cart, or checkout pages.",
        )

        st.number_input(
            "Administrative time (minutes)",
            min_value=0.0,
            max_value=120.0,
            step=0.5,
            key="administrative_duration_min",
        )

        st.number_input(
            "Informational pages",
            min_value=0,
            max_value=30,
            step=1,
            key="informational",
        )

        st.number_input(
            "Informational time (minutes)",
            min_value=0.0,
            max_value=120.0,
            step=0.5,
            key="informational_duration_min",
        )

        st.number_input(
            "Product-related pages",
            min_value=0,
            max_value=450,
            step=1,
            key="product_related",
        )

        st.number_input(
            "Product browsing time (minutes)",
            min_value=0.0,
            max_value=720.0,
            step=1.0,
            key="product_related_duration_min",
        )

    with behaviour_column:
        st.markdown("#### Conversion signals")

        st.slider(
            "Bounce rate",
            min_value=0.0,
            max_value=0.20,
            step=0.001,
            format="%.3f",
            key="bounce_rate",
        )

        st.slider(
            "Exit rate",
            min_value=0.0,
            max_value=0.20,
            step=0.001,
            format="%.3f",
            key="exit_rate",
        )

        st.number_input(
            "Page value",
            min_value=0.0,
            max_value=400.0,
            step=1.0,
            key="page_values",
            help=(
                "A higher value means the viewed pages are more strongly "
                "associated with completed purchases."
            ),
        )

        st.slider(
            "Special-day closeness",
            min_value=0.0,
            max_value=1.0,
            step=0.1,
            key="special_day",
        )

        st.selectbox(
            "Month",
            month_options,
            key="month",
        )

        st.selectbox(
            "Weekend session",
            [False, True],
            format_func=lambda value: "Yes" if value else "No",
            key="weekend",
        )

    with context_column:
        st.markdown("#### Session context")

        st.selectbox(
            "Visitor type",
            visitor_options,
            format_func=lambda value: value.replace("_", " "),
            key="visitor_type",
        )

        st.selectbox(
            "Operating system ID",
            list(range(1, 9)),
            key="operating_system",
            help="Anonymous category ID from the dataset.",
        )

        st.selectbox(
            "Browser ID",
            list(range(1, 14)),
            key="browser",
            help="Anonymous category ID from the dataset.",
        )

        st.selectbox(
            "Region ID",
            list(range(1, 10)),
            key="region",
            help="Anonymous category ID from the dataset.",
        )

        st.selectbox(
            "Traffic source ID",
            list(range(1, 21)),
            key="traffic_type",
            help="Anonymous traffic-source category from the dataset.",
        )

        st.info(
            "Category IDs are labels only. A larger number does not mean "
            "a better browser, region, or traffic source."
        )

    submitted = st.form_submit_button(
        "Analyse purchase intent",
        type="primary",
        use_container_width=True,
    )


# validate the form and run the prediction
if submitted:
    current_inputs = {
        key: st.session_state[key]
        for key in default_inputs
    }

    validation_errors, validation_warnings = validate_inputs(current_inputs)

    if validation_errors:
        st.error("Please correct the following input issue(s):")
        for validation_error in validation_errors:
            st.write(f"- {validation_error}")
    else:
        for validation_warning in validation_warnings:
            st.warning(validation_warning)

        try:
            model_input = build_model_input(
                current_inputs,
                feature_columns,
            )

            probability = predict_probability(
                model,
                model_input,
            )

            if not 0.0 <= probability <= 1.0:
                raise ValueError(
                    "The model returned a probability outside the valid 0 to 1 range."
                )

            timestamp = datetime.now().isoformat(timespec="seconds")

            st.session_state["last_prediction"] = {
                "timestamp": timestamp,
                "scenario": st.session_state["active_preset"],
                "inputs": current_inputs,
                "probability": probability,
                "model_input": model_input,
            }

            st.session_state["prediction_history"].append(
                {
                    "Timestamp": timestamp,
                    "Scenario": st.session_state["active_preset"],
                    "Purchase Probability": probability,
                    "Page Value": current_inputs["page_values"],
                    "Exit Rate": current_inputs["exit_rate"],
                    "Product Pages": current_inputs["product_related"],
                }
            )

        except Exception as error:
            st.error(
                "The prediction could not be completed. Check that the saved "
                "feature columns match the data used to train the model."
            )

            if show_technical:
                st.exception(error)


# display the latest prediction or the empty result state
latest_prediction = st.session_state["last_prediction"]

st.markdown(
    """
    <div class="section-head">
        <div class="section-index">02 / result</div>
        <div>
            <h2>Session decision</h2>
            <p>
                The result combines the predicted purchase probability with the
                decision threshold selected above.
            </p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if latest_prediction is None:
    st.markdown(
        """
        <div class="empty-state">
            <div>
                <h3>No session has been analysed yet.</h3>
                <p>
                    Choose a preset or complete the form, then select
                    <strong>Analyse purchase intent</strong> to view the result.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    probability = float(latest_prediction["probability"])
    predicted_purchase = probability >= threshold
    segment, recommended_action = intent_segment(
        probability,
        threshold,
    )

    outcome = (
        "Likely to purchase"
        if predicted_purchase
        else "Not yet likely to purchase"
    )

    st.markdown(
        f"""
        <div class="result-grid">
            <div class="probability-panel">
                <div>
                    <div class="probability-label">estimated probability</div>
                    <div class="probability-number">{probability:.0%}</div>
                </div>
                <div>
                    <div class="probability-track">
                        <div class="probability-fill" style="width:{probability * 100:.1f}%"></div>
                    </div>
                    <p class="fine-print">
                        model probability for Revenue = 1
                    </p>
                </div>
            </div>
            <div class="decision-panel">
                <div>
                    <div class="probability-label">model decision</div>
                    <h2>{escape(outcome)}</h2>
                    <p>
                        <strong>{escape(segment)}</strong><br>
                        {escape(recommended_action)}
                    </p>
                </div>
                <div class="decision-footer">
                    <div>threshold<br><strong>{threshold:.0%}</strong></div>
                    <div>scenario<br><strong>{escape(latest_prediction['scenario'])}</strong></div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    decision_tab, business_tab, model_tab, history_tab = st.tabs(
        [
            "Decision",
            "Business case",
            "Model view",
            "Session history",
        ]
    )

    # show the recommended action and session signals
    with decision_tab:
        action_column, signal_column = st.columns([0.85, 1.15])

        with action_column:
            st.markdown("### Recommended action")
            st.markdown(
                f'<div class="notice">{escape(recommended_action)}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<p class="fine-print">This is decision-support guidance, '
                'not a guaranteed customer outcome or a causal claim.</p>',
                unsafe_allow_html=True,
            )

        with signal_column:
            st.markdown("### Session signals")

            for signal_title, signal_description in build_business_signals(
                latest_prediction["inputs"]
            ):
                st.markdown(
                    f"""
                    <div class="signal-row">
                        <strong>{escape(signal_title)}</strong>
                        <span>{escape(signal_description)}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # show a simple scenario calculator for the target audience
    with business_tab:
        st.markdown("### Similar-session estimate")
        st.caption(
            "Use this section to translate the model probability into a simple "
            "planning scenario. It is not a guaranteed revenue forecast."
        )

        scenario_column_a, scenario_column_b, scenario_column_c = st.columns(3)

        with scenario_column_a:
            similar_sessions = st.number_input(
                "Number of similar sessions",
                min_value=10,
                max_value=100000,
                value=1000,
                step=100,
            )

        with scenario_column_b:
            average_order_value = st.number_input(
                "Average order value (S$)",
                min_value=1.0,
                max_value=5000.0,
                value=85.0,
                step=5.0,
            )

        with scenario_column_c:
            outreach_cost = st.number_input(
                "Outreach cost per session (S$)",
                min_value=0.0,
                max_value=100.0,
                value=0.20,
                step=0.05,
            )

        expected_purchasers = similar_sessions * probability
        potential_order_value = expected_purchasers * average_order_value
        total_outreach_cost = similar_sessions * outreach_cost

        metric_column_a, metric_column_b, metric_column_c = st.columns(3)
        metric_column_a.metric(
            "Expected purchasers",
            f"{expected_purchasers:,.0f}",
        )
        metric_column_b.metric(
            "Potential gross order value",
            f"S${potential_order_value:,.0f}",
        )
        metric_column_c.metric(
            "Outreach cost",
            f"S${total_outreach_cost:,.0f}",
        )

        st.markdown("### Threshold check")

        threshold_table = pd.DataFrame(
            {
                "Threshold": np.arange(0.20, 0.81, 0.05),
            }
        )
        threshold_table["Predicted class"] = np.where(
            probability >= threshold_table["Threshold"],
            "Purchase",
            "No purchase",
        )
        threshold_table["Session probability"] = probability

        st.dataframe(
            threshold_table.style.format(
                {
                    "Threshold": "{:.0%}",
                    "Session probability": "{:.1%}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    # show model information and global importance when available
    with model_tab:
        st.markdown("### Model summary")

        summary_columns = st.columns(3)
        summary_columns[0].metric("Model", model_name)
        summary_columns[1].metric("Expected features", len(feature_columns))
        summary_columns[2].metric("Saved threshold", f"{saved_threshold:.0%}")

        if isinstance(evaluation_metrics, dict) and evaluation_metrics:
            available_metrics = [
                (key, value)
                for key, value in evaluation_metrics.items()
                if isinstance(value, (int, float))
            ]

            if available_metrics:
                st.markdown("### Saved evaluation results")
                metric_columns = st.columns(min(4, len(available_metrics)))

                for index, (metric_name, metric_value) in enumerate(
                    available_metrics[:4]
                ):
                    metric_columns[index].metric(
                        metric_name,
                        f"{float(metric_value):.4f}",
                    )

        importance_figure = make_feature_importance_chart(
            model,
            feature_columns,
        )

        if importance_figure is not None:
            st.markdown("### Global feature importance")
            st.plotly_chart(
                importance_figure,
                use_container_width=True,
                config={"displayModeBar": False},
            )
            st.caption(
                "Global importance shows what the model generally relies on. "
                "It does not prove that a feature caused this individual result."
            )
        else:
            st.info(
                "The saved model does not expose built-in feature importance."
            )

        if show_technical:
            st.markdown("### Active model inputs")
            active_model_input = latest_prediction["model_input"].T
            active_model_input.columns = ["Value"]
            active_model_input = active_model_input[
                active_model_input["Value"] != 0
            ]
            st.dataframe(
                active_model_input,
                use_container_width=True,
            )

    # show and export predictions created during the current browser session
    with history_tab:
        history_table = pd.DataFrame(
            st.session_state["prediction_history"]
        )

        if history_table.empty:
            st.info("No predictions have been recorded in this session.")
        else:
            display_history = history_table.copy()
            display_history["Purchase Probability"] = display_history[
                "Purchase Probability"
            ].map(lambda value: f"{value:.1%}")

            st.dataframe(
                display_history.iloc[::-1],
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                "Download history as CSV",
                history_table.to_csv(index=False).encode("utf-8"),
                "shopper_prediction_history.csv",
                "text/csv",
                use_container_width=True,
            )

    # create a downloadable record of the latest prediction
    result_payload = {
        "timestamp": latest_prediction["timestamp"],
        "scenario": latest_prediction["scenario"],
        "purchase_probability": probability,
        "decision_threshold": threshold,
        "predicted_purchase": bool(predicted_purchase),
        "intent_segment": segment,
        "recommended_action": recommended_action,
    }

    st.download_button(
        "Download latest result as JSON",
        json.dumps(result_payload, indent=2),
        "shopper_prediction_result.json",
        "application/json",
        use_container_width=True,
    )


# show the application disclaimer and file information
st.markdown(
    f"""
    <div class="footer-note">
        <span>Intent / Commerce — decision support, not a guarantee</span>
        <span>model file: {escape(model_path.name)}</span>
    </div>
    """,
    unsafe_allow_html=True,
)
