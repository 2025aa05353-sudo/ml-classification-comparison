import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# =====================================================
# Page Config
# =====================================================
st.set_page_config(page_title="Wine Quality Classification", layout="wide")

# =====================================================
# Global Header
# =====================================================
st.title("🍷🥂 Wine Quality Classifications for (Red & White Wine) – ML Models Comparison")

# =====================================================
# Load Artifacts
# =====================================================
MODEL_PATH = "model/"

models = {
    "Logistic Regression": joblib.load(MODEL_PATH + "logistic_regression.pkl"),
    "Decision Tree": joblib.load(MODEL_PATH + "decision_tree.pkl"),
    "KNN": joblib.load(MODEL_PATH + "knn.pkl"),
    "Naive Bayes": joblib.load(MODEL_PATH + "naive_bayes.pkl"),
    "Random Forest": joblib.load(MODEL_PATH + "random_forest.pkl"),
    "XGBoost": joblib.load(MODEL_PATH + "xgboost.pkl"),
}

scaler = joblib.load(MODEL_PATH + "scaler.pkl")
label_encoder = joblib.load(MODEL_PATH + "label_encoder.pkl")

X_test = joblib.load(MODEL_PATH + "X_test.pkl")
y_test = joblib.load(MODEL_PATH + "y_test.pkl")
results_df = joblib.load(MODEL_PATH + "results_df.pkl")

df_full = pd.read_csv("data/wine-quality-white-and-red.csv")

# =====================================================
# Prepare results_df safely (FIX)
# =====================================================
if "ML Model Name" in results_df.columns:
    results_df_indexed = results_df.set_index("ML Model Name")
else:
    results_df_indexed = results_df.copy()

# =====================================================
# Constants & Helpers
# =====================================================
EXPECTED_COLUMNS = [
    "type", "fixed acidity", "volatile acidity", "citric acid",
    "residual sugar", "chlorides", "free sulfur dioxide",
    "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"
]

def plot_confusion(y_true, y_pred, title):
    labels = sorted(np.unique(y_true))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="coolwarm",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax
    )
    ax.set_xlabel("Predicted Quality")
    ax.set_ylabel("Actual Quality")
    ax.set_title(title)
    st.pyplot(fig)

# =====================================================
# Sidebar Navigation
# =====================================================
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choose a page",
    ["Model Comparison", "Predict on New Data", "Dataset Info", "Model Info"]
)

# =====================================================
# PAGE 1: Model Comparison
# =====================================================
if page == "Model Comparison":

    st.subheader("Model Performance Comparison")
    results_display = results_df_indexed.reset_index()
    results_display = results_display.rename(columns={"index": "Model Name"})
    results_display.index = range(1, len(results_display) + 1)
    st.dataframe(results_display.round(4))

    st.subheader("Select Model for Detailed Analysis")
    model_name = st.selectbox("Choose a model", list(models.keys()))
    model = models[model_name]

    # ---- FIXED METRICS FETCH ----
    metrics = results_df_indexed.loc[model_name]

    st.markdown(
        f"""
        **Accuracy:** {metrics['Accuracy']:.4f} |
        **Precision:** {metrics['Precision']:.4f} |
        **Recall:** {metrics['Recall']:.4f} |
        **F1:** {metrics['F1']:.4f} |
        **MCC:** {metrics['MCC']:.4f}
        **AUC:** {metrics['AUC']:.4f} |      
        """
    )

    if model_name == "XGBoost":
        y_pred = label_encoder.inverse_transform(model.predict(X_test))
    else:
        y_pred = model.predict(X_test)

    plot_confusion(y_test, y_pred, f"Confusion Matrix – {model_name}")

# =====================================================
# PAGE 2: Predict on New Data
# =====================================================
elif page == "Predict on New Data":

    st.subheader("Predict Wine Quality – upload test Data here and Make Predictions")

    uploaded_file = st.file_uploader(
        "Upload CSV (without quality column)",
        type=["csv"]
    )

    # Sample CSV
    sample_df = pd.DataFrame([
        ["red", 7.4, 0.7, 0.0, 1.9, 0.076, 11, 34, 0.9978, 3.51, 0.56, 9.4],
        ["white", 6.3, 0.3, 0.34, 1.6, 0.049, 14, 132, 0.9940, 3.30, 0.49, 9.5],
        ["red", 7.8, 0.88, 0.0, 2.6, 0.098, 25, 67, 0.9968, 3.20, 0.68, 9.8],
        ["white", 6.2, 0.23, 0.32, 1.6, 0.039, 12, 127, 0.9950, 3.25, 0.50, 9.0],
    ], columns=EXPECTED_COLUMNS)

    st.download_button(
        "Download Sample CSV",
        sample_df.to_csv(index=False),
        file_name="sample_wine_test_data.csv",
        mime="text/csv"
    )

    model_name = st.selectbox("Select Model", list(models.keys()))
    model = models[model_name]

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)

            if list(df.columns) != EXPECTED_COLUMNS:
                st.error("❌ Wrong file uploaded – Please use sample data to update the features for Prediction")
            else:
                df["type"] = df["type"].str.strip().str.lower()
                df["type"] = df["type"].map({"red": 0, "white": 1})
                X_scaled = scaler.transform(df)

                if model_name == "XGBoost":
                    preds = label_encoder.inverse_transform(model.predict(X_scaled))
                else:
                    preds = model.predict(X_scaled)

                df["quality"] = preds
                df["type"] = df["type"].map({0: "red", 1: "white"})

                st.success("Prediction Successful")
                df_display = df.copy()
                # Convert type back to red / white for display
                df_display["type"] = df_display["type"].map({0: "red", 1: "white"})
                # Start index from 1 for display
                df_display.index = range(1, len(df_display) + 1)
                st.dataframe(df_display.head())

                st.download_button(
                    "Download Predicted CSV",
                    df.to_csv(index=False),
                    file_name="predicted_wine_quality.csv",
                    mime="text/csv"
                )
        except Exception:
            st.error("❌ Wrong file uploaded – Please use sample data to update the features for Prediction")

    st.subheader("Expected Data Format (Sample)")
    sample_df_display = sample_df.copy()
    sample_df_display.index = range(1, len(sample_df_display) + 1)
    st.dataframe(sample_df_display)

# =====================================================
# PAGE 3: Dataset Info
# =====================================================
elif page == "Dataset Info":

    st.subheader("Dataset Information")
    st.markdown(
        """
        **Source:** [Kaggle (UCI ML Repository)](https://www.kaggle.com/datasets/ruthgn/wine-quality-data-set-red-white-wine)  
        
        **Target Variable:** quality (3–9)
        
        **Dataset Info:** This data set contains records related to red and white variants of the Portuguese Vinho Verde wine. It contains information from 1599 red wine samples and 4898 white wine samples. Input variables in the data set consist of the type of wine (either red or white wine) and metrics from objective tests (e.g. acidity levels, PH values, ABV, etc.), while the target/output variable is a numerical score based on sensory data—median of at least 3 evaluations made by wine experts.  
        
        **Problem Type:** Multiclass Classification
        """
    )


    # ================================
    # Quick Stats (FROM THE DATA)
    # ================================
    st.markdown("### 📊 Quick Stats")

    num_instances = df_full.shape[0]
    num_features = df_full.shape[1] - 1  # excluding quality
    num_classes = df_full["quality"].nunique()
    quality_min = df_full["quality"].min()
    quality_max = df_full["quality"].max()

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(
            f"""
            - 📊 **Instances:** {num_instances}  
            - 🔢 **Features:** {num_features}  
            - 🎯 **Classes:** {num_classes}  
            - 📈 **Type:** Classification  
            """
        )

    with col2:
        st.info(
            f"""
            **Wine Quality Dataset**

            • Red & White Wines  
            • Quality range: {quality_min} – {quality_max}  
            • Multiclass ML problem
            """
        )


    st.subheader("Feature Information : 13 features")
    feature_info = pd.DataFrame([
        ["Type", "categorical", "Red or White wine"],
        ["Fixed Acidity", "g/dm³", "Non-volatile acids"],
        ["Volatile Acidity", "g/dm³", "Acetic acid amount"],
        ["Citric Acid", "g/dm³", "Adds freshness"],
        ["Residual Sugar", "g/dm³", "Sugar after fermentation"],
        ["Chlorides", "g/dm³", "Salt content"],
        ["Free Sulfur Dioxide", "mg/dm³", "Prevents microbial growth"],
        ["Total Sulfur Dioxide", "mg/dm³", "Total SO₂ concentration"],
        ["Density", "g/cm³", "Wine density"],
        ["pH", "Scale", "Acidity level"],
        ["Sulphates", "g/dm³", "Wine preservative"],
        ["Alcohol", "% vol", "Alcohol content"],
        ["Quality", "score", "Target variable"],
    ], columns=["Feature", "Unit", "Description"])

    feature_info_display = feature_info.copy()
    feature_info_display.index = range(1, len(feature_info_display) + 1)
    st.dataframe(feature_info_display)

# =====================================================
# PAGE 4: Model Info
# =====================================================
else:

    st.subheader("😀 Implemented Models")
    models_info_df = pd.DataFrame([
        ["Logistic Regression", "Linear", "Fast, interpretable"],
        ["Decision Tree", "Tree-based", "Non-linear, interpretable"],
        ["KNN", "Instance-based", "Sensitive to feature scaling"],
        ["Naive Bayes", "Probabilistic", "Fast & simple"],
        ["Random Forest", "Ensemble", "Reduces overfitting"],
        ["XGBoost", "Boosting", "High performance"],
    ], columns=["Model", "Type", "Key Characteristics"])

    models_info_df.index = range(1, len(models_info_df) + 1)

    st.table(models_info_df)

    st.subheader("📏 Evaluation Metrics")

    metrics_info_df = pd.DataFrame([
        ["Accuracy", "Overall correctness", "0 to 1"],
        ["AUC", "Discrimination ability", "0 to 1"],
        ["Precision", "Positive prediction accuracy", "0 to 1"],
        ["Recall", "Coverage of positives", "0 to 1"],
        ["F1 Score", "Balance of precision & recall", "0 to 1"],
        ["MCC", "Balanced correlation metric", "-1 to 1"],
    ], columns=["Metric", "Description", "Range"])

    metrics_info_df.index = range(1, len(metrics_info_df) + 1)

    st.table(metrics_info_df)

