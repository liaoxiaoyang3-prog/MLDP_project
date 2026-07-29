import joblib
import pandas as pd
import streamlit as st


# ============================================================
# PAGE SETUP
# ============================================================

st.set_page_config(
    page_title="Online Shopper Predictor",
    page_icon="🛒",
    layout="wide"
)

st.title("🛒 Online Shopper Purchase Predictor")

st.write(
    "Enter information about an online shopping session "
    "to predict whether the customer is likely to purchase."
)


# ============================================================
# LOAD MODEL
# ============================================================

bundle = joblib.load("online_shopper_model.joblib")

model = bundle["model"]
feature_columns = bundle["feature_columns"]
threshold = bundle.get("threshold", 0.50)


# ============================================================
# USER INPUTS
# ============================================================

st.subheader("Page Activity")

col1, col2, col3 = st.columns(3)

with col1:
    administrative = st.number_input(
        "Administrative pages",
        min_value=0,
        value=2
    )

    administrative_duration = st.number_input(
        "Administrative duration (seconds)",
        min_value=0.0,
        value=80.0
    )

with col2:
    informational = st.number_input(
        "Informational pages",
        min_value=0,
        value=1
    )

    informational_duration = st.number_input(
        "Informational duration (seconds)",
        min_value=0.0,
        value=25.0
    )

with col3:
    product_related = st.number_input(
        "Product-related pages",
        min_value=0,
        value=35
    )

    product_related_duration = st.number_input(
        "Product-related duration (seconds)",
        min_value=0.0,
        value=1200.0
    )


st.subheader("Customer Behaviour")

col4, col5, col6, col7 = st.columns(4)

with col4:
    bounce_rates = st.number_input(
        "Bounce rate",
        min_value=0.0,
        max_value=0.2,
        value=0.01,
        step=0.001,
        format="%.3f"
    )

with col5:
    exit_rates = st.number_input(
        "Exit rate",
        min_value=0.0,
        max_value=0.2,
        value=0.03,
        step=0.001,
        format="%.3f"
    )

with col6:
    page_values = st.number_input(
        "Page value",
        min_value=0.0,
        value=25.0
    )

with col7:
    special_day = st.select_slider(
        "Special-day proximity",
        options=[
            0.0,
            0.2,
            0.4,
            0.6,
            0.8,
            1.0
        ],
        value=0.0
    )


st.subheader("Session Information")

col8, col9, col10 = st.columns(3)

with col8:
    month = st.selectbox(
        "Month",
        [
            "Feb",
            "Mar",
            "May",
            "June",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec"
        ],
        index=8
    )

    visitor_type = st.selectbox(
        "Visitor type",
        [
            "Returning_Visitor",
            "New_Visitor",
            "Other"
        ]
    )

with col9:
    operating_system = st.selectbox(
        "Operating system code",
        list(range(1, 9)),
        index=1
    )

    browser = st.selectbox(
        "Browser code",
        list(range(1, 14)),
        index=1
    )

with col10:
    region = st.selectbox(
        "Region code",
        list(range(1, 10))
    )

    traffic_type = st.selectbox(
        "Traffic type code",
        list(range(1, 21)),
        index=1
    )

weekend = st.radio(
    "Is this a weekend session?",
    ["No", "Yes"],
    horizontal=True
)


# ============================================================
# PREDICTION BUTTON
# ============================================================

if st.button(
    "Predict Purchase",
    type="primary",
    use_container_width=True
):

    # Put user inputs into one row
    input_data = pd.DataFrame([{
        "Administrative": administrative,
        "Administrative_Duration": administrative_duration,
        "Informational": informational,
        "Informational_Duration": informational_duration,
        "ProductRelated": product_related,
        "ProductRelated_Duration": product_related_duration,
        "BounceRates": bounce_rates,
        "ExitRates": exit_rates,
        "PageValues": page_values,
        "SpecialDay": special_day,
        "Month": month,
        "OperatingSystems": operating_system,
        "Browser": browser,
        "Region": region,
        "TrafficType": traffic_type,
        "VisitorType": visitor_type,
        "Weekend": weekend == "Yes"
    }])

    # Columns that need one-hot encoding
    categorical_columns = [
        "Month",
        "OperatingSystems",
        "Browser",
        "Region",
        "TrafficType",
        "VisitorType",
        "Weekend"
    ]

    # Apply one-hot encoding
    input_encoded = pd.get_dummies(
        input_data,
        columns=categorical_columns,
        dtype=int
    )

    # Match the exact columns used during training
    model_input = input_encoded.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # Predict probability
    purchase_probability = model.predict_proba(
        model_input
    )[0, 1]

    # Apply classification threshold
    prediction = int(
        purchase_probability >= threshold
    )

    st.divider()
    st.subheader("Prediction Result")

    result1, result2 = st.columns(2)

    with result1:
        st.metric(
            "Purchase probability",
            f"{purchase_probability:.1%}"
        )

    with result2:
        st.metric(
            "Model prediction",
            "Purchase" if prediction == 1 else "No Purchase"
        )

    if prediction == 1:
        st.success(
            "The customer is likely to make a purchase."
        )
    else:
        st.warning(
            "The customer is unlikely to make a purchase."
        )

    with st.expander("View submitted information"):
        st.dataframe(
            input_data,
            use_container_width=True,
            hide_index=True
        )