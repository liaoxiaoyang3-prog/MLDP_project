# import required libraries

from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import sklearn
import streamlit as st


# configure page

st.set_page_config(
    page_title="Model Test",
    layout="centered"
)

st.title("Model Loading Test")


# show deployment environment

st.subheader("Deployment environment")

st.write("Python:", sys.version)
st.write("scikit-learn:", sklearn.__version__)
st.write("joblib:", joblib.__version__)
st.write("numpy:", np.__version__)
st.write("pandas:", pd.__version__)


# show files available in github deployment

app_folder = Path(__file__).resolve().parent

st.subheader("Files found")

files_found = [
    file.name
    for file in app_folder.iterdir()
]

st.write(files_found)


# set exact model filename

model_path = (
    app_folder
    / "online_shopper_model.joblib"
)

st.write("Model path:", str(model_path))
st.write("Model exists:", model_path.exists())


# attempt to load model

st.subheader("Model loading result")

try:
    loaded_object = joblib.load(
        model_path
    )

    st.success("The model file loaded successfully.")

    st.write(
        "Loaded object type:",
        str(type(loaded_object))
    )

    if isinstance(loaded_object, dict):
        st.write(
            "Bundle keys:",
            list(loaded_object.keys())
        )

        if "model" in loaded_object:
            st.write(
                "Model type:",
                str(type(loaded_object["model"]))
            )

        if "feature_columns" in loaded_object:
            st.write(
                "Number of features:",
                len(
                    loaded_object[
                        "feature_columns"
                    ]
                )
            )

except Exception as error:
    st.error("The model failed to load.")

    # show the real error
    st.exception(error)