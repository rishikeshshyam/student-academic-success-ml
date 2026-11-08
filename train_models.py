"""
Train and compare five classification models for the Student Academic Success dataset.

The training and test CSV files have already been prepared.
This script fits models on train_data.csv and evaluates them on test_data.csv.
"""

from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


# Files and folders used by this project
PROJECT_FOLDER = Path(__file__).resolve().parent
TRAIN_FILE = PROJECT_FOLDER / "train_data.csv"
TEST_FILE = PROJECT_FOLDER / "test_data.csv"

MODEL_FOLDER = PROJECT_FOLDER / "model"
RESULT_FOLDER = PROJECT_FOLDER / "results"

TARGET = "Target"
CLASS_NAMES = ["Dropout", "Enrolled", "Graduate"]


# These columns contain category codes, so they should be one-hot encoded.
categorical_columns = [
    "Marital status",
    "Application mode",
    "Course",
    "Daytime/evening attendance",
    "Previous qualification",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "International",
]

# These columns represent measurements, grades, counts or ordered values.
numeric_columns = [
    "Application order",
    "Previous qualification (grade)",
    "Admission grade",
    "Age at enrollment",
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (without evaluations)",
    "Unemployment rate",
    "Inflation rate",
    "GDP",
]


def create_preprocessor(model_name):
    """Prepare category and numeric columns before model training."""

    category_steps = Pipeline(steps=[
        ("fill_missing", SimpleImputer(strategy="most_frequent")),
        ("one_hot", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )),
    ])

    number_steps = Pipeline(steps=[
        ("fill_missing", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])

    # Gaussian Naive Bayes assumes continuous features. One-hot columns are
    # binary 0/1 values, which break that assumption and made the model
    # predict almost every student as "Enrolled" (accuracy 0.244).
    # So Naive Bayes uses only the 19 numeric columns.
    if model_name == "Naive Bayes":
        return ColumnTransformer(transformers=[
            ("numbers", number_steps, numeric_columns),
        ])

    preprocessor = ColumnTransformer(transformers=[
        ("categories", category_steps, categorical_columns),
        ("numbers", number_steps, numeric_columns),
    ])

    return preprocessor


def create_models():
    """Set up the five classifiers required for the assignment."""

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "Support Vector Machine": SVC(
            kernel="rbf",
            probability=True,
            random_state=42,
        ),
    }

    return models


def main():
    # Read the two files created during the train/test split.
    train_data = pd.read_csv(TRAIN_FILE)
    test_data = pd.read_csv(TEST_FILE)

    print("Training data shape:", train_data.shape)
    print("Testing data shape:", test_data.shape)

    # Check that every feature is included in one of the two lists.
    feature_columns = categorical_columns + numeric_columns

    if len(feature_columns) != 36 or len(set(feature_columns)) != 36:
        raise ValueError("Check the feature lists. There should be 36 unique features.")

    if TARGET not in train_data.columns or TARGET not in test_data.columns:
        raise ValueError("Both CSV files must contain the Target column.")

    if set(feature_columns) != set(train_data.columns) - {TARGET}:
        raise ValueError("The feature lists do not match the training CSV columns.")

    if set(train_data.columns) != set(test_data.columns):
        raise ValueError("Training and testing CSV files have different columns.")

    # Separate the features from the target.
    X_train = train_data[feature_columns]
    y_train = train_data[TARGET]

    X_test = test_data[feature_columns]
    y_test = test_data[TARGET]

    print("\nTraining target counts:")
    print(y_train.value_counts())

    print("\nTesting target counts:")
    print(y_test.value_counts())

    # Create folders for the trained models and evaluation results.
    MODEL_FOLDER.mkdir(exist_ok=True)
    RESULT_FOLDER.mkdir(exist_ok=True)

    # Keep each model's metrics so they can be compared at the end.
    all_results = []

    for model_name, classifier in create_models().items():
        print("\nTraining", model_name)

        # Preprocessing is part of the pipeline, so it is fitted only on training data.
        model = Pipeline(steps=[
            ("preprocessing", create_preprocessor(model_name)),
            ("classifier", classifier),
        ])

        model.fit(X_train, y_train)

        # Predict the outcome for the separate test records.
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)

        # Use the model's class order when calculating multiclass AUC.
        model_classes = list(model.named_steps["classifier"].classes_)
        probability_data = pd.DataFrame(probabilities, columns=model_classes)
        probabilities = probability_data[CLASS_NAMES].to_numpy()

        accuracy = accuracy_score(y_test, predictions)
        auc = roc_auc_score(
            y_test,
            probabilities,
            labels=CLASS_NAMES,
            multi_class="ovr",
            average="macro",
        )
        precision = precision_score(
            y_test, predictions, labels=CLASS_NAMES,
            average="macro", zero_division=0
        )
        recall = recall_score(
            y_test, predictions, labels=CLASS_NAMES,
            average="macro", zero_division=0
        )
        f1 = f1_score(
            y_test, predictions, labels=CLASS_NAMES,
            average="macro", zero_division=0
        )
        mcc = matthews_corrcoef(y_test, predictions)

        # Store the six metrics for the final comparison table.
        all_results.append({
            "Model": model_name,
            "Accuracy": accuracy,
            "AUC": auc,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "MCC": mcc,
        })

        # Save the complete pipeline, including preprocessing.
        file_name = model_name.lower().replace(" ", "_")
        joblib.dump(model, MODEL_FOLDER / f"{file_name}.joblib")

        # Save the confusion matrix and class-wise report for this model.
        matrix = confusion_matrix(
            y_test, predictions, labels=CLASS_NAMES
        )
        matrix_table = pd.DataFrame(
            matrix,
            index=CLASS_NAMES,
            columns=CLASS_NAMES,
        )
        matrix_table.index.name = "Actual"
        matrix_table.columns.name = "Predicted"
        matrix_table.to_csv(RESULT_FOLDER / f"{file_name}_confusion_matrix.csv")

        report = classification_report(
            y_test,
            predictions,
            labels=CLASS_NAMES,
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        )
        pd.DataFrame(report).transpose().to_csv(
            RESULT_FOLDER / f"{file_name}_classification_report.csv"
        )

        print("Accuracy:", round(accuracy, 4))
        print("AUC:", round(auc, 4))
        print("Precision:", round(precision, 4))
        print("Recall:", round(recall, 4))
        print("F1:", round(f1, 4))
        print("MCC:", round(mcc, 4))

    # Save all model results in one comparison table.
    results_table = pd.DataFrame(all_results)
    results_table = results_table.set_index("Model")
    results_table.to_csv(RESULT_FOLDER / "model_comparison.csv")

    print("\nFinal comparison of the five models:")
    print(results_table.round(4))

    print("\nTraining and evaluation are complete.")
    print("Trained models are saved in:", MODEL_FOLDER)
    print("Evaluation results are saved in:", RESULT_FOLDER)


if __name__ == "__main__":
    main()
