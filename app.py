import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from src.utils import setup_logging

# --- Path Constants ---
PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
TRAINING_RESULTS_PATH = MODELS_DIR / "training_results.json"

# Model options mapping label to filename
MODEL_OPTIONS = {
    "Logistic Regression": "logistic_regression",
    "Random Forest": "random_forest",
    "Decision Tree": "decision_tree",
}

# Features expected by the models (must match training order)
FEATURE_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]

logger = setup_logging("streamlit_app")


def setup_ui():
    """Configures the Streamlit page layout."""
    st.set_page_config(
        page_title="Heart Disease Risk Predictor",
        page_icon="❤️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Minimal CSS — only button styling, theme colors come from .streamlit/config.toml
    st.markdown(
        """
        <style>
            .stButton>button {
                background-color: #4f8ef7;
                color: white;
                border-radius: 5px;
            }
            .stButton>button:hover {
                background-color: #3b7de0;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_model(model_name: str):
    """Loads a pre-trained model from a pickle file."""
    model_path = MODELS_DIR / f"{model_name}.pkl"
    if not model_path.exists():
        st.error(
            f"Model file not found: {model_path}.\nTrain models first: python src/main.py --train"
        )
        return None
    try:
        return joblib.load(model_path)
    except Exception as exc:
        st.error(f"Failed to load model {model_name}: {exc}")
        return None


@st.cache_data
def load_training_results():
    """Loads the evaluation metrics generated during training."""
    if not TRAINING_RESULTS_PATH.exists():
        return None
    try:
        with TRAINING_RESULTS_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        st.error(f"Failed to load training results: {exc}")
        return None


def main():
    """Main entry point for the Streamlit application."""
    setup_ui()
    st.title("❤️ Heart Disease Risk Predictor")
    st.markdown(
        "Use this interface to evaluate heart disease risk with trained machine learning models."
    )
    st.sidebar.title("Navigation")

    page = st.sidebar.radio(
        "Select a page",
        ["Home", "Prediction", "Model Performance", "About"],
    )
    if page == "Home":
        show_home()
    elif page == "Prediction":
        show_prediction()
    elif page == "Model Performance":
        show_performance()
    else:
        show_about()


def show_home():
    """Displays the landing page."""
    st.header("Welcome")
    st.write(
        "This project predicts the likelihood of heart disease using clinical features from the UCI dataset. "
        "Use the CLI or the Streamlit interface depending on your workflow."
    )
    st.subheader("How to use this app")
    st.markdown(
        """
        1. Train models using the CLI: `python src/main.py --train`
        2. Start Streamlit from the CLI: `python src/main.py --streamlit`
        3. Choose a model and enter patient data.
        4. Click **Make Prediction** to see the risk assessment.
        """
    )
    st.info(
        "The predictions are educational only and not a medical diagnosis. Always consult a healthcare professional."
    )


def get_patient_features():
    """Displays the input form and returns the feature list."""
    st.subheader("Patient parameters")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider("Age", 29, 77, 55)
        sex = st.selectbox("Sex", ["Female", "Male"])
        cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
        trestbps = st.slider("Resting Blood Pressure", 90, 200, 130)

    with col2:
        chol = st.slider("Cholesterol", 125, 565, 240)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
        restecg = st.selectbox("Resting ECG result", [0, 1, 2])
        thalach = st.slider("Max Heart Rate", 60, 202, 150)

    with col3:
        exang = st.selectbox("Exercise Induced Angina", [0, 1])
        oldpeak = st.slider("ST Depression", 0.0, 6.2, 1.0, step=0.1)
        slope = st.selectbox("ST Slope", [0, 1, 2])
        ca = st.selectbox("Number of major vessels (ca)", [0, 1, 2, 3])
        thal = st.selectbox("Thalassemia", [0, 1, 2, 3])

    return [
        age,
        1 if sex == "Male" else 0,
        cp, trestbps, chol, fbs, restecg,
        thalach, exang, oldpeak, slope, ca, thal,
    ]


def show_prediction():
    """Interactive form for entering patient data and generating predictions."""
    st.header("Risk Prediction")
    st.write("Select a model and input patient metrics to estimate heart disease probability.")

    model_label = st.selectbox("Prediction Model", list(MODEL_OPTIONS.keys()))

    st.markdown("---")
    input_values = get_patient_features()

    if st.button("Make Prediction"):
        model = load_model(MODEL_OPTIONS[model_label])
        if model is None:
            return

        input_df = pd.DataFrame([input_values], columns=FEATURE_COLUMNS)
        prediction = model.predict(input_df)[0]
        probability = get_prediction_probability(model, input_df)
        display_prediction_results(prediction, probability)


def get_prediction_probability(model, input_df):
    """Extracts probability or decision score from the model."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(input_df)[0][1]
    if hasattr(model, "decision_function"):
        return model.decision_function(input_df)[0]
    return float("nan")


def display_prediction_results(prediction, probability):
    """Renders the prediction outcome to the UI."""
    st.markdown("---")
    st.subheader("Prediction Analysis")

    res_col1, _ = st.columns(2)
    with res_col1:
        risk_text = "High risk of heart disease" if prediction == 1 else "Low risk of heart disease"
        risk_icon = "⚠️" if prediction == 1 else "✅"
        st.metric("Risk assessment", f"{risk_icon} {risk_text}")
        st.write(f"Probability of disease: {probability * 100:.1f}%")

        if prediction == 1:
            st.warning(
                "The model predicts a higher probability of heart disease. This is for educational purposes only."
            )
        else:
            st.success("The model predicts a lower probability of heart disease.")


def show_performance():
    """Visualizes model metrics from training_results.json."""
    st.header("Model Performance")
    results = load_training_results()
    if results is None:
        st.error("No training results found. Train models first using the CLI.")
        return

    df = pd.DataFrame(results).T.round(4)
    st.subheader("Comparison Table")
    st.dataframe(df, use_container_width=True)

    st.markdown("---")
    st.write("Model comparison charts based on training results.")

    col1, col2 = st.columns(2)
    with col1:
        st.bar_chart(df["accuracy"])
    with col2:
        st.bar_chart(df["roc_auc"])

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.bar_chart(df["precision"])
    with col2:
        st.bar_chart(df["recall"])
    with col3:
        st.bar_chart(df["f1"])


def show_about():
    """Information about the dataset and project."""
    st.header("About")
    st.markdown(
        """
        ### Heart Disease Risk Predictor

        This application demonstrates a machine learning workflow for predicting heart disease using a clinical dataset.

        - **Models:** Logistic Regression, Random Forest, Decision Tree
        - **Dataset:** UCI Heart Disease dataset
        - **Goal:** Evaluate risk and explain how predictions are produced

        ### Important notes
        - This is **not** a medical diagnostic tool.
        - It is intended for learning and demonstration.
        - Always consult a healthcare professional for medical decisions.
        """
    )


if __name__ == "__main__":
    main()