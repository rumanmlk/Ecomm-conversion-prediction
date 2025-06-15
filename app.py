# app.py

# 1. Set page config first
import streamlit as st
st.set_page_config(page_title="E-commerce Conversion Predictor", layout="wide")

# 2. Import libraries
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 3. Load models and feature names
@st.cache_resource
def load_models():
    rf_pipeline = joblib.load("models/full_pipeline.joblib")
    feature_names = joblib.load("models/feature_names.joblib")
    nn_model = tf.keras.models.load_model("models/final_nn_model.keras")
    return rf_pipeline, feature_names, nn_model

rf_pipeline, feature_names, nn_model = load_models()

# 4. Helper functions
def preprocess_input(input_df):
    """
    Align input DataFrame with the feature names used during training.
    Missing columns are filled with zeros, and extra columns are dropped.
    """
    # Ensure all required columns are present
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0
    # Drop any extra columns
    input_df = input_df[feature_names]
    return input_df

def predict(input_df):
    """
    Generate predictions using both Random Forest and Neural Network models.
    Returns probabilities and predicted classes.
    """
    # Preprocess input
    processed_input = preprocess_input(input_df.copy())

    # Random Forest predictions
    rf_probs = rf_pipeline.predict_proba(processed_input)[:, 1]
    rf_preds = (rf_probs >= 0.5).astype(int)

    # Neural Network predictions
    nn_probs = nn_model.predict(processed_input).ravel()
    nn_preds = (nn_probs >= 0.5).astype(int)

    return rf_probs, rf_preds, nn_probs, nn_preds

# 5. Streamlit UI
st.title("🛒 E-commerce Conversion Predictor")
st.markdown("Predict the likelihood of a user making a purchase based on session features.")

# Sidebar for input method selection
input_method = st.sidebar.radio("Select Input Method:", ["Manual Entry", "Upload CSV"])

if input_method == "Manual Entry":
    st.sidebar.subheader("Input Features")

    # Example manual inputs (adjust based on your feature set)
    feature_inputs = {}
    for feature in feature_names:
        # For simplicity, using number inputs; customize as needed
        feature_inputs[feature] = st.sidebar.number_input(f"{feature}", value=0.0)

    input_df = pd.DataFrame([feature_inputs])

    if st.button("Predict"):
        rf_probs, rf_preds, nn_probs, nn_preds = predict(input_df)

        st.subheader("Prediction Results")
        results_df = pd.DataFrame({
            "Model": ["Random Forest", "Neural Network"],
            "Probability": [rf_probs[0], nn_probs[0]],
            "Prediction": ["Purchase" if rf_preds[0] == 1 else "No Purchase",
                           "Purchase" if nn_preds[0] == 1 else "No Purchase"]
        })
        st.dataframe(results_df)

        # Plotting probabilities
        st.subheader("Prediction Probabilities")
        fig, ax = plt.subplots()
        models = ["Random Forest", "Neural Network"]
        probabilities = [rf_probs[0], nn_probs[0]]
        sns.barplot(x=models, y=probabilities, ax=ax)
        ax.set_ylabel("Probability of Purchase")
        ax.set_ylim(0, 1)
        st.pyplot(fig)

elif input_method == "Upload CSV":
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)
            st.write("Uploaded Data:")
            st.dataframe(input_df.head())

            if st.button("Predict"):
                rf_probs, rf_preds, nn_probs, nn_preds = predict(input_df)

                st.subheader("Prediction Results")
                results_df = input_df.copy()
                results_df["RF_Probability"] = rf_probs
                results_df["RF_Prediction"] = ["Purchase" if pred == 1 else "No Purchase" for pred in rf_preds]
                results_df["NN_Probability"] = nn_probs
                results_df["NN_Prediction"] = ["Purchase" if pred == 1 else "No Purchase" for pred in nn_preds]
                st.dataframe(results_df)

                # Plotting average probabilities
                st.subheader("Average Prediction Probabilities")
                avg_rf_prob = np.mean(rf_probs)
                avg_nn_prob = np.mean(nn_probs)
                fig, ax = plt.subplots()
                models = ["Random Forest", "Neural Network"]
                probabilities = [avg_rf_prob, avg_nn_prob]
                sns.barplot(x=models, y=probabilities, ax=ax)
                ax.set_ylabel("Average Probability of Purchase")
                ax.set_ylim(0, 1)
                st.pyplot(fig)

        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")
