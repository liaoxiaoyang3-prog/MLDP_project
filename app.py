from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="IntentIQ | Shopper Purchase Intelligence",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "Online_shopper_intention_GB_model"

DEFAULT_MODEL_METRICS = {
    "F2 Score": 0.7571,
    "Recall": 0.8325,
    "ROC-AUC": 0.9274,
    "Average Precision": 0.7246,
}

MONTH_OPTIONS = [
    "Feb", "Mar", "May", "June", "Jul",
    "Aug", "Sep", "Oct", "Nov", "Dec"
]

VISITOR_OPTIONS = [
    "Returning_Visitor",
    "New_Visitor",
    "Other",
]

DEFAULT_INPUTS = {
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

PRESETS = {
    "High-intent shopper": {
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
    "Product explorer": {
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
    "At-risk visitor": {
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

st.markdown(
    """
    <style>
        :root {
            --ink: #11233f;
            --muted: #60708a;
            --surface: rgba(255, 255, 255, 0.92);
            --line: rgba(41, 76, 122, 0.14);
        }
        .stApp {
            background:
                radial-gradient(circle at 10% 0%, rgba(108, 92, 231, 0.11), transparent 28%),
                radial-gradient(circle at 92% 5%, rgba(0, 184, 148, 0.10), transparent 24%),
                #f7f9fd;
        }
        .block-container {max-width: 1450px; padding-top: 1.4rem; padding-bottom: 4rem;}
        [data-testid="stSidebar"] {background: linear-gradient(180deg, #111c35 0%, #182847 100%);}
        [data-testid="stSidebar"] * {color: #f5f7ff;}
        [data-testid="stSidebar"] .stSlider label,
        [data-testid="stSidebar"] .stCheckbox label {color: #f5f7ff !important;}
        .hero {
            position: relative; overflow: hidden; padding: 2.15rem 2.25rem;
            border-radius: 26px; color: white;
            background: linear-gradient(120deg, rgba(18,31,61,.98), rgba(74,60,173,.95));
            box-shadow: 0 22px 55px rgba(36,48,92,.20); margin-bottom: 1.35rem;
        }
        .hero:after {content:""; position:absolute; width:360px; height:360px; right:-110px; top:-170px; border-radius:50%; background:rgba(255,255,255,.10);}
        .eyebrow {display:inline-block; font-size:.76rem; font-weight:800; letter-spacing:.16em; text-transform:uppercase; padding:.38rem .72rem; border-radius:999px; background:rgba(255,255,255,.12); border:1px solid rgba(255,255,255,.16); margin-bottom:.85rem;}
        .hero h1 {color:white; margin:0; font-size:clamp(2rem,4vw,3.8rem); letter-spacing:-.055em; line-height:1.02;}
        .hero p {margin:.95rem 0 0; max-width:820px; color:rgba(255,255,255,.80); font-size:1.02rem; line-height:1.65;}
        .result-card {border-radius:22px; padding:1.35rem 1.45rem; color:white; box-shadow:0 18px 42px rgba(38,55,96,.18);}
        .result-hot {background:linear-gradient(135deg,#5f4bd8,#7f73ea);}
        .result-warm {background:linear-gradient(135deg,#008f72,#00b894);}
        .result-cool {background:linear-gradient(135deg,#355070,#52769d);}
        .result-card h2 {color:white; margin:.2rem 0 .35rem; font-size:2rem;}
        .result-card p {margin:0; color:rgba(255,255,255,.84); line-height:1.55;}
        .mini-label {font-size:.74rem; letter-spacing:.12em; text-transform:uppercase; font-weight:800; opacity:.76;}
        .signal {border-left:4px solid #6c5ce7; background:white; border-radius:12px; padding:.85rem 1rem; margin:.55rem 0; border:1px solid rgba(41,76,122,.10); border-left-width:4px;}
        .signal strong {color:#172442;}
        .empty-state {text-align:center; padding:3.2rem 1.2rem; border:1px dashed rgba(67,88,130,.28); border-radius:22px; background:rgba(255,255,255,.65);}
        .empty-state .icon {font-size:2.7rem; margin-bottom:.5rem;}
        div[data-testid="stMetric"] {background:rgba(255,255,255,.92); border:1px solid rgba(41,76,122,.12); border-radius:16px; padding:.85rem 1rem; box-shadow:0 8px 24px rgba(35,58,95,.06);}
        div[data-testid="stForm"] {background:rgba(255,255,255,.80); border:1px solid rgba(41,76,122,.12); border-radius:22px; padding:1rem 1.1rem 1.15rem; box-shadow:0 12px 35px rgba(35,58,95,.07);}
        .stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {border-radius:12px; font-weight:750; min-height:2.75rem;}
        .section-title {margin-top:.2rem; margin-bottom:.2rem; font-size:1.35rem; color:#172442; font-weight:800; letter-spacing:-.025em;}
        .section-copy {color:#66758d; margin-bottom:.9rem;}
        .fine-print {color:#78869b; font-size:.82rem; line-height:1.5;}
        /* Fix sidebar button text */
        [data-testid="stSidebar"] .stButton > button {
            background-color: #FFFFFF !important;
            color: #172442 !important;
            border: 1px solid #AAB4C5 !important;
            font-weight: 700 !important;
        }

        [data-testid="stSidebar"] .stButton > button p,
        [data-testid="stSidebar"] .stButton > button span {
            color: #172442 !important;
            opacity: 1 !important;
        }

        [data-testid="stSidebar"] .stButton > button:hover {
            background-color: #E8E5FF !important;
            color: #4B3CC4 !important;
            border-color: #6C5CE7 !important;
        }

        [data-testid="stSidebar"] .stButton > button:hover p,
        [data-testid="stSidebar"] .stButton > button:hover span {
            color: #4B3CC4 !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model_bundle(path: Path) -> dict[str, Any]:
    bundle = joblib.load(path)
    required_keys = {"model", "feature_columns"}
    missing = required_keys.difference(bundle.keys())
    if missing:
        raise KeyError("The model bundle is missing: " + ", ".join(sorted(missing)))
    if not isinstance(bundle["feature_columns"], list):
        raise TypeError("'feature_columns' must be stored as a list.")
    return bundle


def initialise_session_state() -> None:
    for key, value in DEFAULT_INPUTS.items():
        st.session_state.setdefault(key, value)
    st.session_state.setdefault("active_preset", "Custom session")
    st.session_state.setdefault("last_prediction", None)
    st.session_state.setdefault("prediction_history", [])


def apply_preset(preset_name: str) -> None:
    for key, value in PRESETS[preset_name].items():
        st.session_state[key] = value
    st.session_state["active_preset"] = preset_name
    st.session_state["last_prediction"] = None
    st.rerun()


def set_if_present(frame: pd.DataFrame, feature_name: str, value: float | int) -> None:
    if feature_name in frame.columns:
        frame.at[0, feature_name] = value


def set_category(frame: pd.DataFrame, base_name: str, value: str | int | bool) -> None:
    columns = set(frame.columns)
    if base_name in columns:
        if isinstance(value, bool):
            frame.at[0, base_name] = int(value)
        elif isinstance(value, (int, float)):
            frame.at[0, base_name] = value

    candidates = [f"{base_name}_{value}", f"{base_name}_{str(value)}"]
    if isinstance(value, bool):
        candidates.extend([f"{base_name}_{int(value)}", f"{base_name}_{str(value).lower()}"])
    for candidate in candidates:
        if candidate in columns:
            frame.at[0, candidate] = 1.0


def build_model_input(inputs: dict[str, Any], feature_columns: list[str]) -> pd.DataFrame:
    row = pd.DataFrame(np.zeros((1, len(feature_columns))), columns=feature_columns)

    admin_duration = inputs["administrative_duration_min"] * 60.0
    info_duration = inputs["informational_duration_min"] * 60.0
    product_duration = inputs["product_related_duration_min"] * 60.0

    numerical_values = {
        "Administrative": inputs["administrative"],
        "Administrative_Duration": admin_duration,
        "Informational": inputs["informational"],
        "Informational_Duration": info_duration,
        "ProductRelated": inputs["product_related"],
        "ProductRelated_Duration": product_duration,
        "BounceRates": inputs["bounce_rate"],
        "ExitRates": inputs["exit_rate"],
        "PageValues": inputs["page_values"],
        "SpecialDay": inputs["special_day"],
    }
    for name, value in numerical_values.items():
        set_if_present(row, name, value)

    engineered = {
        "AvgAdministrativeTime": admin_duration / inputs["administrative"] if inputs["administrative"] else 0.0,
        "AvgInformationalTime": info_duration / inputs["informational"] if inputs["informational"] else 0.0,
        "AvgProductTime": product_duration / inputs["product_related"] if inputs["product_related"] else 0.0,
        "TotalPages": inputs["administrative"] + inputs["informational"] + inputs["product_related"],
        "TotalDuration": admin_duration + info_duration + product_duration,
        "VisitedAdministrative": int(inputs["administrative"] > 0),
        "VisitedInformational": int(inputs["informational"] > 0),
        "VisitedProductPage": int(inputs["product_related"] > 0),
        "HasPageValue": int(inputs["page_values"] > 0),
    }
    for name, value in engineered.items():
        set_if_present(row, name, value)

    set_category(row, "Month", inputs["month"])
    set_category(row, "Weekend", inputs["weekend"])
    set_category(row, "OperatingSystems", inputs["operating_system"])
    set_category(row, "Browser", inputs["browser"])
    set_category(row, "Region", inputs["region"])
    set_category(row, "TrafficType", inputs["traffic_type"])
    set_category(row, "VisitorType", inputs["visitor_type"])
    return row


def validate_inputs(inputs: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    pairs = [
        ("administrative", "administrative_duration_min", "Administrative"),
        ("informational", "informational_duration_min", "Informational"),
        ("product_related", "product_related_duration_min", "Product-related"),
    ]
    for count_key, duration_key, label in pairs:
        if inputs[count_key] == 0 and inputs[duration_key] > 0:
            errors.append(f"{label} time cannot be above zero when the page count is zero.")
    return errors


def predict_probability(model: Any, model_input: pd.DataFrame) -> float:
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(model_input)[0, 1])
    if hasattr(model, "decision_function"):
        score = float(model.decision_function(model_input)[0])
        return float(1 / (1 + np.exp(-score)))
    raise TypeError("The saved model does not support probability predictions.")


def intent_segment(probability: float, threshold: float) -> tuple[str, str, str]:
    if probability >= 0.75:
        return "High-intent shopper", "result-hot", "Prioritise personalised recommendations and a timely checkout reminder."
    if probability >= threshold:
        return "Likely purchaser", "result-warm", "Use a gentle conversion nudge, product reassurance, or cart reminder."
    if probability >= 0.25:
        return "Consideration stage", "result-cool", "Show relevant products, reviews, and useful information without over-targeting."
    return "Low immediate intent", "result-cool", "Avoid expensive promotions; focus on helpful browsing and future re-engagement."


def build_business_signals(inputs: dict[str, Any]) -> list[tuple[str, str]]:
    signals: list[tuple[str, str]] = []
    if inputs["page_values"] >= 20:
        signals.append(("Strong conversion signal", "PageValues is high, one of the strongest indicators found during EDA."))
    elif inputs["page_values"] == 0:
        signals.append(("Limited conversion signal", "PageValues is zero, so the session has not yet shown a strong purchase-linked page pattern."))
    else:
        signals.append(("Developing conversion signal", "PageValues is above zero but still moderate."))

    if inputs["exit_rate"] <= 0.04:
        signals.append(("Low exit risk", "The exit rate is relatively low, suggesting the visitor is continuing through the site."))
    elif inputs["exit_rate"] >= 0.12:
        signals.append(("High exit risk", "The exit rate is high, so the visitor may leave before completing a purchase."))

    if inputs["product_related"] >= 40:
        signals.append(("Deep product exploration", "The visitor viewed many product-related pages."))
    elif inputs["product_related"] <= 5:
        signals.append(("Limited product engagement", "The visitor viewed only a small number of product-related pages."))

    if inputs["visitor_type"] == "Returning_Visitor":
        signals.append(("Returning visitor", "The shopper has returned to the website, which can support stronger follow-up actions."))
    return signals[:4]


def make_probability_gauge(probability: float, threshold: float) -> go.Figure:
    figure = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        number={"suffix": "%", "font": {"size": 52, "color": "#172442"}},
        title={"text": "Estimated purchase probability", "font": {"size": 18, "color": "#60708a"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#718096"},
            "bar": {"color": "#6c5ce7", "thickness": 0.30},
            "bgcolor": "white",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 25], "color": "#edf2f7"},
                {"range": [25, 50], "color": "#dfe8f5"},
                {"range": [50, 75], "color": "#c8f3e7"},
                {"range": [75, 100], "color": "#ddd8ff"},
            ],
            "threshold": {"line": {"color": "#e17055", "width": 4}, "thickness": 0.78, "value": threshold * 100},
        },
    ))
    figure.update_layout(height=320, margin=dict(l=20, r=20, t=55, b=10), paper_bgcolor="rgba(0,0,0,0)")
    return figure


def make_feature_importance_chart(model: Any, feature_columns: list[str]) -> go.Figure | None:
    if hasattr(model, "feature_importances_"):
        importance = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        importance = np.abs(np.asarray(model.coef_)[0])
    else:
        return None
    if len(importance) != len(feature_columns):
        return None
    table = pd.DataFrame({"Feature": feature_columns, "Importance": importance})
    table = table.sort_values("Importance", ascending=False).head(12).sort_values("Importance")
    figure = go.Figure(go.Bar(
        x=table["Importance"], y=table["Feature"], orientation="h",
        marker={"color": table["Importance"], "colorscale": "Purples", "showscale": False},
        hovertemplate="%{y}<br>Importance: %{x:.4f}<extra></extra>",
    ))
    figure.update_layout(height=430, margin=dict(l=10, r=15, t=20, b=20), xaxis_title="Global feature importance", yaxis_title="", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return figure


initialise_session_state()

st.markdown(
    """
    <div class="hero">
        <h1>IntentIQ</h1>
        <p>Turn a live shopping session into an estimated purchase probability, a clear customer-intent segment, and an actionable marketing response. Built for e-commerce and digital marketing teams.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not MODEL_PATH.exists():
    st.error("The model file was not found. Place `Online_shopper_intention_GB_model` in the same folder as `app.py`.")
    st.code("shopper_intention_app/\n├── app.py\n├── Online_shopper_intention_GB_model\n├── requirements.txt\n└── .streamlit/\n    └── config.toml")
    st.stop()

try:
    bundle = load_model_bundle(MODEL_PATH)
except Exception as exc:
    st.error("The model bundle could not be loaded.")
    st.exception(exc)
    st.stop()

model = bundle["model"]
feature_columns = bundle["feature_columns"]
saved_threshold = float(bundle.get("threshold", 0.50))
model_name = bundle.get("model_name", model.__class__.__name__)
evaluation_metrics = bundle.get("evaluation_metrics", DEFAULT_MODEL_METRICS)

if hasattr(model, "feature_names_in_") and list(model.feature_names_in_) != feature_columns:
    st.error("The saved feature-column order does not match the model. Export the bundle again using the exact training columns.")
    st.stop()

with st.sidebar:
    st.markdown("## 🛍️ IntentIQ")
    st.caption("Shopper purchase-intention intelligence")
    st.markdown("---")
    st.markdown("### Decision settings")
    threshold = st.slider(
        "Purchase decision threshold",
        min_value=0.10,
        max_value=0.90,
        value=float(np.clip(saved_threshold, 0.10, 0.90)),
        step=0.01,
        help="The probability does not change. This only changes when the session is labelled Purchase.",
    )
    st.caption("Lower threshold → more purchasers identified, but usually more false positives.")
    show_technical = st.checkbox("Show technical details", value=False)
    st.markdown("---")
    st.markdown("### Model status")
    st.success("Model loaded")
    st.write(f"**Model:** {model_name}")
    st.write(f"**Expected features:** {len(feature_columns)}")
    st.write(f"**Saved threshold:** {saved_threshold:.2f}")
    if st.button("Clear latest analysis", use_container_width=True):
        st.session_state["last_prediction"] = None
        st.rerun()

st.markdown('<div class="section-title">Start with a demo shopper</div>', unsafe_allow_html=True)
st.markdown('<div class="section-copy">Load a ready-made scenario or enter a custom session below.</div>', unsafe_allow_html=True)
preset_columns = st.columns(3)
for column, preset_name in zip(preset_columns, PRESETS):
    with column:
        if st.button(preset_name, key=f"preset_{preset_name}", use_container_width=True):
            apply_preset(preset_name)
st.caption(f"Active scenario: **{st.session_state['active_preset']}**")

with st.form("shopper_session_form", clear_on_submit=False):
    st.markdown("### Session behaviour")
    engagement_col, behaviour_col, context_col = st.columns(3)

    with engagement_col:
        st.markdown("#### Page engagement")
        st.number_input("Administrative pages", 0, 30, step=1, key="administrative", help="Examples include account, login, cart, or checkout pages.")
        st.number_input("Administrative time (minutes)", 0.0, 120.0, step=0.5, key="administrative_duration_min")
        st.number_input("Informational pages", 0, 30, step=1, key="informational")
        st.number_input("Informational time (minutes)", 0.0, 120.0, step=0.5, key="informational_duration_min")
        st.number_input("Product-related pages", 0, 450, step=1, key="product_related")
        st.number_input("Product browsing time (minutes)", 0.0, 720.0, step=1.0, key="product_related_duration_min")

    with behaviour_col:
        st.markdown("#### Conversion signals")
        st.slider("Bounce rate", 0.0, 0.20, step=0.001, format="%.3f", key="bounce_rate")
        st.slider("Exit rate", 0.0, 0.20, step=0.001, format="%.3f", key="exit_rate")
        st.number_input("Page value", 0.0, 400.0, step=1.0, key="page_values", help="Higher values mean the pages are more associated with purchases.")
        st.slider("Special-day closeness", 0.0, 1.0, step=0.1, key="special_day")
        st.selectbox("Month", MONTH_OPTIONS, key="month")
        st.selectbox("Weekend session", [False, True], format_func=lambda v: "Yes" if v else "No", key="weekend")

    with context_col:
        st.markdown("#### Session context")
        st.selectbox("Visitor type", VISITOR_OPTIONS, format_func=lambda v: v.replace("_", " "), key="visitor_type")
        st.selectbox("Operating system ID", list(range(1, 9)), key="operating_system", help="Anonymous category ID.")
        st.selectbox("Browser ID", list(range(1, 14)), key="browser", help="Anonymous category ID.")
        st.selectbox("Region ID", list(range(1, 10)), key="region", help="Anonymous region ID.")
        st.selectbox("Traffic source ID", list(range(1, 21)), key="traffic_type", help="Anonymous traffic-source category.")
        st.info("The category IDs are anonymous and should not be interpreted as rankings.")

    submitted = st.form_submit_button("Analyse shopper intent", type="primary", use_container_width=True)

if submitted:
    current_inputs = {key: st.session_state[key] for key in DEFAULT_INPUTS}
    errors = validate_inputs(current_inputs)
    if errors:
        for error in errors:
            st.error(error)
    else:
        try:
            model_input = build_model_input(current_inputs, feature_columns)
            probability = predict_probability(model, model_input)
            timestamp = datetime.now().isoformat(timespec="seconds")
            st.session_state["last_prediction"] = {
                "timestamp": timestamp,
                "scenario": st.session_state["active_preset"],
                "inputs": current_inputs,
                "probability": probability,
                "model_input": model_input,
            }
            st.session_state["prediction_history"].append({
                "Timestamp": timestamp,
                "Scenario": st.session_state["active_preset"],
                "Purchase Probability": probability,
                "Page Value": current_inputs["page_values"],
                "Exit Rate": current_inputs["exit_rate"],
                "Product Pages": current_inputs["product_related"],
            })
        except Exception as exc:
            st.error("Prediction failed. Check that the saved feature columns match the model training data.")
            if show_technical:
                st.exception(exc)

latest = st.session_state["last_prediction"]

if latest is None:
    st.markdown(
        """
        <div class="empty-state">
            <div class="icon">✨</div>
            <h3>Your prediction dashboard is ready</h3>
            <p>Choose a demo shopper or enter a session, then press <strong>Analyse shopper intent</strong>.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    probability = float(latest["probability"])
    predicted_purchase = probability >= threshold
    segment, result_class, action = intent_segment(probability, threshold)

    st.markdown("---")
    st.markdown('<div class="section-title">Shopper intelligence result</div>', unsafe_allow_html=True)
    gauge_col, decision_col = st.columns([1.05, 0.95], vertical_alignment="center")

    with gauge_col:
        st.plotly_chart(make_probability_gauge(probability, threshold), use_container_width=True, config={"displayModeBar": False})

    with decision_col:
        outcome = "Likely to purchase" if predicted_purchase else "Not yet likely to purchase"
        st.markdown(
            f"""
            <div class="result-card {result_class}">
                <div class="mini-label">Model decision</div>
                <h2>{outcome}</h2>
                <p><strong>{segment}</strong><br>{action}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        metric_a, metric_b = st.columns(2)
        metric_a.metric("Purchase probability", f"{probability:.1%}")
        metric_b.metric("Decision threshold", f"{threshold:.0%}")

    result_tab, strategy_tab, insights_tab, history_tab = st.tabs([
        "🎯 Recommended action",
        "📈 Strategy lab",
        "🧠 Model insights",
        "🕘 Session history",
    ])

    with result_tab:
        action_col, signals_col = st.columns([0.92, 1.08])
        with action_col:
            st.markdown("### Recommended next action")
            if probability >= 0.75:
                st.success("High priority: show personalised products, reduce checkout friction, and send a timely cart reminder. Avoid automatically discounting a shopper who already appears ready to buy.")
            elif probability >= threshold:
                st.success("Conversion opportunity: provide reviews, delivery information, or a gentle checkout reminder.")
            elif probability >= 0.25:
                st.info("Nurture the session: improve product discovery and save items for later rather than using an aggressive promotion.")
            else:
                st.warning("Low immediate intent: avoid expensive targeting. Focus on useful content and future re-engagement.")
            st.markdown('<p class="fine-print">These recommendation rules are business guidance layered on top of the model probability. They are not causal claims.</p>', unsafe_allow_html=True)
        with signals_col:
            st.markdown("### Session signals")
            for title, explanation in build_business_signals(latest["inputs"]):
                st.markdown(f'<div class="signal"><strong>{title}</strong><br><span>{explanation}</span></div>', unsafe_allow_html=True)

    with strategy_tab:
        st.markdown("### Similar-session scenario")
        strategy_a, strategy_b, strategy_c = st.columns(3)
        with strategy_a:
            similar_sessions = st.number_input("Number of similar sessions", 10, 100000, value=1000, step=100)
        with strategy_b:
            average_order_value = st.number_input("Average order value (S$)", 1.0, 5000.0, value=85.0, step=5.0)
        with strategy_c:
            outreach_cost = st.number_input("Outreach cost per session (S$)", 0.0, 100.0, value=0.20, step=0.05)

        expected_purchasers = similar_sessions * probability
        potential_gross_value = expected_purchasers * average_order_value
        total_outreach_cost = similar_sessions * outreach_cost
        a, b, c = st.columns(3)
        a.metric("Expected purchasers", f"{expected_purchasers:,.0f}")
        b.metric("Potential gross order value", f"S${potential_gross_value:,.0f}")
        c.metric("Outreach cost", f"S${total_outreach_cost:,.0f}")
        st.caption("Scenario estimate only. This is not a causal uplift or guaranteed revenue forecast.")

        st.markdown("### Threshold effect")
        threshold_table = pd.DataFrame({"Threshold": np.arange(0.20, 0.81, 0.05)})
        threshold_table["Predicted Class"] = np.where(probability >= threshold_table["Threshold"], "Purchase", "No Purchase")
        threshold_table["Probability"] = probability
        st.dataframe(threshold_table.style.format({"Threshold": "{:.0%}", "Probability": "{:.1%}"}), use_container_width=True, hide_index=True)

    with insights_tab:
        i1, i2, i3, i4 = st.columns(4)
        i1.metric("Notebook F2", f"{evaluation_metrics.get('F2 Score', 0):.4f}")
        i2.metric("Notebook recall", f"{evaluation_metrics.get('Recall', 0):.2%}")
        i3.metric("Notebook ROC-AUC", f"{evaluation_metrics.get('ROC-AUC', 0):.4f}")
        i4.metric("Average precision", f"{evaluation_metrics.get('Average Precision', 0):.4f}")
        st.caption("These are saved or fallback notebook evaluation results. Update the bundle whenever the model is retrained.")

        importance_figure = make_feature_importance_chart(model, feature_columns)
        if importance_figure is not None:
            st.markdown("### Global feature importance")
            st.plotly_chart(importance_figure, use_container_width=True, config={"displayModeBar": False})
            st.caption("Global importance does not prove that a feature caused this individual prediction.")
        else:
            st.info("This model does not expose built-in feature importance.")

        if show_technical:
            st.markdown("### Active model inputs")
            active_input = latest["model_input"].T
            active_input.columns = ["Value"]
            active_input = active_input[active_input["Value"] != 0]
            st.dataframe(active_input, use_container_width=True)

    with history_tab:
        history = pd.DataFrame(st.session_state["prediction_history"])
        if history.empty:
            st.info("No predictions have been recorded in this session.")
        else:
            display_history = history.copy()
            display_history["Purchase Probability"] = display_history["Purchase Probability"].map(lambda v: f"{v:.1%}")
            st.dataframe(display_history.iloc[::-1], use_container_width=True, hide_index=True)
            st.download_button("Download session history as CSV", history.to_csv(index=False).encode("utf-8"), "shopper_intent_history.csv", "text/csv", use_container_width=True)

    result_payload = {
        "timestamp": latest["timestamp"],
        "scenario": latest["scenario"],
        "purchase_probability": probability,
        "decision_threshold": threshold,
        "predicted_purchase": bool(predicted_purchase),
        "intent_segment": segment,
        "recommended_action": action,
    }
    st.download_button("Download latest result as JSON", json.dumps(result_payload, indent=2), "shopper_intent_result.json", "application/json")

st.markdown("---")
st.markdown('<p class="fine-print">IntentIQ produces an estimated purchase probability, not a guarantee. Monitor performance, fairness, drift, and changing customer behaviour.</p>', unsafe_allow_html=True)