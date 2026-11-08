"""Streamlit app for the Student Academic Success classification project."""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)


# Project folders and target information
ROOT = Path(__file__).resolve().parent
MODEL_FOLDER = ROOT / "model"
RESULTS_FOLDER = ROOT / "results"

TARGET = "Target"
CLASSES = ["Dropout", "Enrolled", "Graduate"]

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}


def load_model(model_name):
    """Load a model that was trained earlier by train_models.py."""

    model_path = MODEL_FOLDER / MODEL_FILES[model_name]

    if not model_path.exists():
        raise FileNotFoundError(
            f"{model_path.name} is missing. Run train_models.py first."
        )

    return joblib.load(model_path)


def get_metrics(model, X, y):
    """Predict the test records and calculate the six assignment metrics."""

    predictions = model.predict(X)

    # Arrange predicted probabilities in the same order as the target classes.
    model_classes = list(model.named_steps["classifier"].classes_)
    probability_table = pd.DataFrame(
        model.predict_proba(X),
        columns=model_classes,
    )
    probabilities = probability_table[CLASSES].to_numpy()

    scores = {
        "Accuracy": accuracy_score(y, predictions),
        "AUC": roc_auc_score(
            y,
            probabilities,
            labels=CLASSES,
            multi_class="ovr",
            average="macro",
        ),
        "Precision": precision_score(
            y, predictions, average="macro", zero_division=0
        ),
        "Recall": recall_score(
            y, predictions, average="macro", zero_division=0
        ),
        "F1 Score": f1_score(
            y, predictions, average="macro", zero_division=0
        ),
        "MCC": matthews_corrcoef(y, predictions),
    }

    matrix = confusion_matrix(y, predictions, labels=CLASSES)

    report = classification_report(
        y,
        predictions,
        labels=CLASSES,
        target_names=CLASSES,
        output_dict=True,
        zero_division=0,
    )

    return scores, matrix, report


# Page heading
st.set_page_config(
    page_title="Student Outcome Classification",
    page_icon="🎓",
    layout="wide",
)

st.title("Student Academic Outcome Classification")
st.write(
    "Upload the held-out test dataset, select a trained classifier, "
    "and review its performance."
)
st.caption(
    "This application loads saved models. It does not train or refit "
    "models using the uploaded test data."
)

# Select a model and upload the test CSV.
model_name = st.selectbox("Select a classification model", list(MODEL_FILES))

uploaded_file = st.file_uploader(
    "Upload test_data.csv",
    type=["csv"],
)

if uploaded_file is None:
    st.info("Upload the test CSV file to begin evaluation.")
    st.stop()

# Read the uploaded file.
try:
    data = pd.read_csv(uploaded_file)
except Exception as error:
    st.error(f"Could not read the CSV file: {error}")
    st.stop()

# Check the target column.
if TARGET not in data.columns:
    st.error("The uploaded file must contain the Target column.")
    st.stop()

if data.empty or data[TARGET].isna().any():
    st.error("The uploaded file is empty or has missing target values.")
    st.stop()

unknown_classes = set(data[TARGET].astype(str)) - set(CLASSES)
if unknown_classes:
    st.error(f"Unexpected target values: {sorted(unknown_classes)}")
    st.stop()

# Load the selected model to get the feature names used during training.
try:
    model = load_model(model_name)
except Exception as error:
    st.error(str(error))
    st.stop()

X = data.drop(columns=[TARGET])
y = data[TARGET]

expected_columns = list(model.feature_names_in_)
missing_columns = sorted(set(expected_columns) - set(X.columns))
extra_columns = sorted(set(X.columns) - set(expected_columns))

if missing_columns or extra_columns:
    st.error(
        f"Feature columns do not match the training data. "
        f"Missing: {missing_columns}. Extra: {extra_columns}."
    )
    st.stop()

# Keep columns in the same order as model training.
X = X[expected_columns]

# Show a short summary of the test data.
st.subheader("Test dataset")
col1, col2, col3 = st.columns(3)
col1.metric("Records", f"{len(data):,}")
col2.metric("Input features", len(X.columns))
col3.metric("Target classes", y.nunique())

with st.expander("View target distribution"):
    st.dataframe(
        y.value_counts().reindex(CLASSES, fill_value=0).rename("Records"),
        use_container_width=True,
    )

# Evaluate the selected model when the user clicks the button.
if st.button("Evaluate selected model", type="primary"):
    try:
        scores, matrix, report = get_metrics(model, X, y)

        st.subheader(f"{model_name} - Evaluation metrics")

        first_row = st.columns(3)
        second_row = st.columns(3)

        for column, metric in zip(first_row, list(scores.items())[:3]):
            column.metric(metric[0], f"{metric[1]:.4f}")

        for column, metric in zip(second_row, list(scores.items())[3:]):
            column.metric(metric[0], f"{metric[1]:.4f}")

        st.subheader("Confusion matrix")
        st.dataframe(
            pd.DataFrame(matrix, index=CLASSES, columns=CLASSES),
            use_container_width=True,
        )

        st.subheader("Classification report")
        st.dataframe(
            pd.DataFrame(report).transpose().round(4),
            use_container_width=True,
        )

    except Exception as error:
        st.error(f"Evaluation could not be completed: {error}")


# Optional comparison of all five saved models on the same uploaded test file.
if st.button("Compare all five models"):
    comparison_rows = []

    try:
        for name in MODEL_FILES:
            saved_model = load_model(name)
            scores, _, _ = get_metrics(saved_model, X, y)
            comparison_rows.append({"Model": name, **scores})

        comparison = pd.DataFrame(comparison_rows).set_index("Model")

        st.subheader("Comparison of all five models")
        st.dataframe(
            comparison.style.format("{:.4f}"),
            use_container_width=True,
        )

        st.download_button(
            "Download comparison as CSV",
            comparison.to_csv().encode("utf-8"),
            file_name="model_comparison_from_app.csv",
            mime="text/csv",
        )

    except Exception as error:
        st.error(f"Model comparison could not be completed: {error}")


# Show the saved training comparison, if it is available.
comparison_file = RESULTS_FOLDER / "model_comparison.csv"

if comparison_file.exists():
    with st.expander("View results saved during model training"):
        saved_results = pd.read_csv(comparison_file, index_col=0)
        st.dataframe(saved_results.round(4), use_container_width=True)
