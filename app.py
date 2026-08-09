"""
ML Assignment 2 - Streamlit App
Breast Cancer Wisconsin Classification
Student ID: 2025AC05601
"""

import json
import os

import joblib
import pandas as pd
import streamlit as st
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

st.set_page_config(
    page_title="ML Assignment 2",
    layout="wide",
    initial_sidebar_state="expanded",
)

STUDENT_ID = "2025AC05601"
TARGET_COLUMN = "target"
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl",
}

# basic html/css (beginner level)
st.markdown(
    """
    <style>
    h1 {
        font-size: 22px;
        color: #0d47a1;
        background-color: #bbdefb;
        padding: 8px 10px;
        border: 1px solid #64b5f6;
    }
    h2.compare {
        font-size: 18px;
        color: #1b5e20;
        background-color: #dcedc8;
        padding: 6px 10px;
        margin-top: 18px;
        border: 1px solid #81c784;
    }
    h2.model {
        font-size: 18px;
        color: #4a148c;
        background-color: #e1bee7;
        padding: 6px 10px;
        margin-top: 18px;
        border: 1px solid #ba68c8;
    }
    h2.model .model-name { color: #1565c0; font-weight: bold; }
    h3 {
        font-size: 16px;
        color: #e65100;
        background-color: #ffe0b2;
        padding: 5px 8px;
        border: 1px solid #ffb74d;
    }
    p { font-size: 14px; color: #444; }
    .box {
        border: 1px solid #aaa;
        padding: 12px;
        margin-bottom: 15px;
        background-color: #fff;
    }
    .box-title.metrics { color: #c62828; font-weight: bold; font-size: 15px; }
    .box-title.cm { color: #1565c0; font-weight: bold; font-size: 15px; }
    .box-title.report { color: #2e7d32; font-weight: bold; font-size: 15px; }
    table {
        border-collapse: collapse;
        width: 100%;
        font-size: 14px;
    }
    th, td {
        border: 1px solid #999;
        padding: 6px 8px;
        text-align: center;
    }
    th { background-color: #e8e8e8; }
    section[data-testid="stSidebar"] {
        background-color: #f0f0f0;
    }
    section[data-testid="stSidebar"] h3 {
        font-size: 16px;
        margin-bottom: 8px;
        color: #e65100;
        background-color: #ffe0b2;
        padding: 5px 8px;
        border: 1px solid #ffb74d;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_feature_names():
    with open(os.path.join(MODEL_DIR, "feature_names.json")) as f:
        return json.load(f)


@st.cache_resource
def load_model(model_name):
    return joblib.load(os.path.join(MODEL_DIR, MODEL_FILES[model_name]))


def compute_metrics(y_true, y_pred, y_prob):
    return {
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "AUC": round(roc_auc_score(y_true, y_prob), 4),
        "Precision": round(precision_score(y_true, y_pred), 4),
        "Recall": round(recall_score(y_true, y_pred), 4),
        "F1 Score": round(f1_score(y_true, y_pred), 4),
        "MCC": round(matthews_corrcoef(y_true, y_pred), 4),
    }


def dataframe_to_html_table(df):
    html = "<table><tr>"
    for col in df.columns:
        html += f"<th>{col}</th>"
    html += "</tr>"
    for row in df.itertuples(index=False):
        html += "<tr>"
        for value in row:
            html += f"<td>{value}</td>"
        html += "</tr>"
    html += "</table>"
    return html


def metrics_to_html_table(metrics):
    html = "<table><tr><th>Metric</th><th>Value</th></tr>"
    for name, value in metrics.items():
        html += f"<tr><td>{name}</td><td>{value}</td></tr>"
    html += "</table>"
    return html


def confusion_matrix_to_html_table(cm):
    html = """
    <table>
    <tr>
        <th>Actual \\ Predicted</th>
        <th>Malignant (0)</th>
        <th>Benign (1)</th>
    </tr>
    """
    labels = ["Malignant (0)", "Benign (1)"]
    for i, label in enumerate(labels):
        html += f"<tr><th>{label}</th><td>{cm[i][0]}</td><td>{cm[i][1]}</td></tr>"
    html += "</table>"
    return html


# ----- fixed sidebar -----
st.sidebar.markdown(
    """
    <h3>ML Assignment 2</h3>
    <p>Student ID: 2025AC05601</p>
    <hr>
    <p><b>1.</b> Select model</p>
    """,
    unsafe_allow_html=True,
)

selected_model = st.sidebar.selectbox("Model", list(MODEL_FILES.keys()), label_visibility="collapsed")

st.sidebar.markdown("<p><b>2.</b> Upload test CSV</p>", unsafe_allow_html=True)
uploaded_file = st.sidebar.file_uploader("CSV file", type=["csv"], label_visibility="collapsed")

# ----- main page -----
st.markdown(
    f"""
    <div class="box">
        <h1>Breast Cancer Classification</h1>
        <p>This app compares 5 ML models on breast cancer test data.</p>
        <p><b>Student ID:</b> {STUDENT_ID}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    data_label = uploaded_file.name

    feature_names = load_feature_names()
    missing_cols = [c for c in feature_names + [TARGET_COLUMN] if c not in df.columns]
    if missing_cols:
        st.error(f"Missing columns: {missing_cols}")
        st.stop()

    X = df[feature_names]
    y_true = df[TARGET_COLUMN]

    st.markdown(
        f"""
        <div class="box">
            <p><b>Test data loaded:</b> {data_label}</p>
            <p><b>Number of rows:</b> {len(df)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    comparison_rows = []
    for model_name in MODEL_FILES:
        model = load_model(model_name)
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]
        row = compute_metrics(y_true, y_pred, y_prob)
        row["Model"] = model_name
        comparison_rows.append(row)

    comparison_df = pd.DataFrame(comparison_rows)[
        ["Model", "Accuracy", "AUC", "Precision", "Recall", "F1 Score", "MCC"]
    ]

    st.markdown(
        """
        <h2 class="compare">All Models Comparison</h2>
        <div class="box">
        """
        + dataframe_to_html_table(comparison_df)
        + "</div>",
        unsafe_allow_html=True,
    )

    model = load_model(selected_model)
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    metrics = compute_metrics(y_true, y_pred, y_prob)
    cm = confusion_matrix(y_true, y_pred)

    report_df = pd.DataFrame(
        classification_report(
            y_true,
            y_pred,
            target_names=["Malignant (0)", "Benign (1)"],
            output_dict=True,
        )
    ).transpose().round(4)

    st.markdown(
        f"""
        <h2 class="model">Selected Model: <span class="model-name">{selected_model}</span></h2>
        <div class="box">
            <p class="box-title metrics">Evaluation Metrics</p>
            {metrics_to_html_table(metrics)}
        </div>
        <div class="box">
            <p class="box-title cm">Confusion Matrix</p>
            {confusion_matrix_to_html_table(cm)}
        </div>
        <div class="box">
            <p class="box-title report">Classification Report</p>
            {dataframe_to_html_table(report_df.reset_index().rename(columns={"index": "Class"}))}
        </div>
        """,
        unsafe_allow_html=True,
    )

else:
    st.markdown(
        """
        <div class="box">
            <p>Please upload a test CSV file using the sidebar.</p>
            <p>Use <b>test_data.csv</b> from this project folder.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
