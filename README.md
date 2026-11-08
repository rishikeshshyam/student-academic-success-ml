# Student Academic Performance Classification

A multiclass machine learning project that compares five classification algorithms to classify students' academic outcomes as **Dropout**, **Enrolled**, or **Graduate**. The project covers data preparation, model training, evaluation, saved model pipelines, and a Streamlit application for evaluating held-out test data.

## a. Problem Statement

The objective of this project is to build and compare five machine learning classification models using student-related information from the UCI *Predict Students' Dropout and Academic Success* dataset.

The target variable is `Target`, which contains three possible outcomes:

- `Dropout`
- `Enrolled`
- `Graduate`

The models use application details, demographic and socioeconomic information, academic records, and macroeconomic variables to classify the recorded student outcome.

**Scope of prediction:** The dataset includes academic performance information from both the first and second semesters. Therefore, this project classifies the eventual academic outcome using the available enrollment and semester information. It should not be interpreted as an admission-time or pre-enrollment dropout prediction system.

## b. Dataset Description

**Dataset name:** Predict Students' Dropout and Academic Success  
**Source:** UCI Machine Learning Repository  
**Dataset ID:** 697  
**Official dataset page:** <https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success>  
**DOI:** <https://doi.org/10.24432/C5MC89>  
**License:** Creative Commons Attribution 4.0 International (CC BY 4.0)

The dataset contains records from a higher education institution and includes student application information, demographic and socioeconomic attributes, and academic performance.

| Dataset detail | Value |
|---|---:|
| Total instances | 4,424 |
| Input features | 36 |
| Target column | `Target` |
| Number of target classes | 3 |
| Training records | 3,539 |
| Testing records | 885 |
| Training/testing split | 80:20, stratified |

### Target class distribution

The following distribution is from the complete 4,424-row dataset used to prepare the training and test files.

| Target class | Records | Percentage |
|---|---:|---:|
| Graduate | 2,209 | 49.93% |
| Dropout | 1,421 | 32.12% |
| Enrolled | 794 | 17.95% |
| **Total** | **4,424** | **100.00%** |

The target classes are not equally represented. For this reason, the model comparison includes macro-averaged Precision, Recall and F1, in addition to Accuracy, AUC and MCC.

### Feature preprocessing

The 36 input features are handled in two groups:

- **17 categorical features:** integer-coded categories such as course, application mode, previous qualification, nationality, parents' qualification and occupation, and binary attributes.
- **19 numerical or ordinal features:** grades, age, application order, curricular-unit counts, unemployment rate, inflation rate and GDP.

The preprocessing is included in Scikit-learn Pipelines:

- For Logistic Regression, Decision Tree, KNN and SVM, categorical features are imputed using the most frequent value and one-hot encoded. Numerical features are median-imputed and standard-scaled.
- For Gaussian Naive Bayes, the implementation uses the 19 numerical features with median imputation and standard scaling. The categorical features are not supplied to this model. This is a modelling choice made for this experiment, so its results should be interpreted with that difference in input features in mind.
- Preprocessing steps are fitted on the training data only and then applied to the held-out test data.

The supplied CSV data was checked before modelling. The complete dataset had no missing cells or duplicate rows in the checks performed for this project.

## c. GitHub Repository Link

**GitHub repository:** <https://github.com/rishikeshshyam/student-academic-success-ml>

The repository contains the source code, requirements file, README, held-out test CSV, saved models, and evaluation outputs.

## d. Models Used and Evaluation

The following five classification algorithms were trained using `train_data.csv` and evaluated using the separate `test_data.csv` file:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

### Evaluation metrics

- **Accuracy:** proportion of correctly classified test records.
- **AUC:** macro-averaged multiclass area under the ROC curve using the One-vs-Rest approach.
- **Precision:** macro-averaged precision across the three classes.
- **Recall:** macro-averaged recall across the three classes.
- **F1 Score:** macro-averaged F1 score across the three classes.
- **MCC:** Matthews Correlation Coefficient, calculated for the multiclass predictions.

### Model comparison table

The values below are from the successful BITS Virtual Lab execution of `train_models.py` on 8 November 2026. Values are rounded to four decimal places.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7819 | 0.8876 | 0.7371 | 0.6956 | 0.7068 | 0.6382 |
| Decision Tree | 0.6655 | 0.7187 | 0.6074 | 0.6092 | 0.6078 | 0.4620 |
| KNN | 0.7040 | 0.7956 | 0.6508 | 0.6146 | 0.6242 | 0.5052 |
| Gaussian Naive Bayes | 0.6904 | 0.7803 | 0.6560 | 0.6309 | 0.6336 | 0.4978 |
| Support Vector Machine | 0.7819 | 0.8872 | 0.7411 | 0.6950 | 0.7088 | 0.6393 |

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | It achieved an accuracy of 0.7819 and the highest AUC of 0.8876. Its macro F1 score was 0.7068. It performed strongly overall, although the Enrolled class remains more difficult to classify than the other outcomes. |
| Decision Tree | It achieved an accuracy of 0.6655, an AUC of 0.7187 and a macro F1 score of 0.6078. These were the lowest Accuracy, AUC, F1 and MCC values among the five models in this experiment. |
| KNN | It achieved an accuracy of 0.7040 and an AUC of 0.7956. Its macro F1 score was 0.6242, placing it above the Decision Tree but below Logistic Regression and SVM on this test set. |
| Gaussian Naive Bayes | It achieved an accuracy of 0.6904, an AUC of 0.7803 and a macro F1 score of 0.6336. Its performance was between KNN and the two strongest models on several metrics. It used only the 19 numerical features, unlike the other four models, which used all 36 features. This difference should be considered when comparing the results. |
| Support Vector Machine | It achieved an accuracy of 0.7819, the highest macro Precision (0.7411), macro F1 (0.7088) and MCC (0.6393). Its AUC of 0.8872 was very close to Logistic Regression. It was the strongest model on the selected macro F1 and MCC measures, although the difference from Logistic Regression was small. |
| **Overall winner for this dataset** | **Support Vector Machine** is selected based on its highest macro F1, macro Precision and MCC. Logistic Regression achieved the highest AUC and had very similar overall performance. The difference between these two models is small. The Enrolled class remains a challenge for the models, so class-wise performance should be considered along with the aggregate metrics. |

### Saved evaluation files

After running `train_models.py`, the project saves:

- `results/model_comparison.csv` — comparison of all five models.
- `results/*_confusion_matrix.csv` — confusion matrix for each model.
- `results/*_classification_report.csv` — class-wise precision, recall and F1 report for each model.

Five fitted preprocessing-and-classifier pipelines are saved in the `model/` folder as Joblib files.

## Streamlit Application

The Streamlit application is saved as `app.py`.

The application provides:

- CSV upload for held-out test data.
- A dropdown to select one of the five trained models.
- Display of Accuracy, AUC, Precision, Recall, F1 and MCC.
- Confusion matrix and classification report.
- Option to compare all five saved models on the uploaded test data.

The application loads the saved pipelines from the `model/` folder. It does not retrain the models or fit preprocessing steps on uploaded test data. The uploaded CSV must contain all 36 input features and the `Target` column.

**BITS Virtual Lab execution:** Completed successfully on 8 November 2026.  
**Streamlit application:** Deployed on Streamlit Community Cloud.

**GitHub repository:** <https://github.com/rishikeshshyam/student-academic-success-ml>  
**Live Streamlit Community Cloud URL:** <https://student-academic-success-ml-esmwbysnbxtatxxx3snic3.streamlit.app/>

## How to Run the Project

Run the following commands from the project folder in Windows PowerShell.

### 1. Create and activate a virtual environment

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install the required packages

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Train and evaluate the models

The project already contains the prepared `train_data.csv` and `test_data.csv` files. The training script uses the training file for fitting and the test file for final evaluation. It does not create a second split.

```powershell
python train_models.py
```

### 4. Run the Streamlit application

```powershell
python -m streamlit run app.py
```

Upload `test_data.csv` in the application and select a model to view its evaluation.

## Project Structure

The main project files are organized as follows:

```text
Student Academic Success/
├── app.py
├── train_models.py
├── requirements.txt
├── README.md
├── train_data.csv
├── test_data.csv
├── model/
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── naive_bayes.joblib
│   └── support_vector_machine.joblib
├── results/
│   ├── model_comparison.csv
│   ├── *_confusion_matrix.csv
│   └── *_classification_report.csv
└── Dataset/
    └── Student_Academic_Success.xlsx
```

## Dataset Reference

Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2021). *Predict Students' Dropout and Academic Success* [Dataset]. UCI Machine Learning Repository. <https://doi.org/10.24432/C5MC89>.

Dataset license: Creative Commons Attribution 4.0 International (CC BY 4.0).
