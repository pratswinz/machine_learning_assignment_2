"""
ML Assignment 2 - Streamlit App
Breast Cancer Wisconsin Classification
Student ID: 2025AC05601 | BITS WILP M.Tech AIML/DSE
"""

import json
import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
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
    roc_curve,
)

# --- Page setup ---
st.set_page_config(
    page_title="2025AC05601 | ML Assignment 2",
    page_icon="🎗️",
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

if "use_sample" not in st.session_state:
    st.session_state.use_sample = True
if "data_source" not in st.session_state:
    st.session_state.data_source = None

# basic styling
st.markdown(
    """
    <style>
    .winner-badge {
        background: #ecfdf5;
        color: #047857;
        padding: 0.2rem 0.6rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 700;
        border: 1px solid #6ee7b7;
    }
    div[data-testid="stSidebar"] {
        background-color: #f8fafc;
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


def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="PuBu",
        linewidths=0.5,
        xticklabels=["Malignant (0)", "Benign (1)"],
        yticklabels=["Malignant (0)", "Benign (1)"],
        ax=ax,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"{model_name}\nConfusion Matrix", fontsize=11, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_roc_curve(y_true, y_prob, model_name):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_val = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, color="#2563eb", lw=2, label=f"AUC = {auc_val:.4f}")
    ax.plot([0, 1], [0, 1], "--", color="#94a3b8", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{model_name}\nROC Curve", fontsize=11, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def plot_metrics_bars(metrics_dict, model_name):
    names = list(metrics_dict.keys())
    values = list(metrics_dict.values())
    colors = ["#2563eb" if n != "MCC" else "#7c3aed" for n in names]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    bars = ax.barh(names, values, color=colors, edgecolor="white")
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Score")
    ax.set_title(f"Metric Scores — {model_name}", fontsize=11, fontweight="bold")
    for bar, val in zip(bars, values):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    return fig


st.title("Breast Cancer Tumor Classification")
st.caption(f"ML Assignment 2 · {STUDENT_ID}")

# ===== SIDEBAR =====
selected_model = st.sidebar.selectbox("Model", list(MODEL_FILES.keys()))

uploaded_file = st.sidebar.file_uploader("Upload test CSV", type=["csv"])

col_a, col_b = st.sidebar.columns(2)
with col_a:
    if st.button("Load Sample", use_container_width=True):
        if os.path.exists(os.path.join(os.path.dirname(__file__), "test_data.csv")):
            st.session_state.use_sample = True
            st.session_state.data_source = "sample"
        else:
            st.sidebar.error("test_data.csv missing from repo.")
with col_b:
    if st.button("Clear", use_container_width=True):
        st.session_state.use_sample = False
        st.session_state.data_source = None

# ===== MAIN =====
if uploaded_file is not None or st.session_state.get("use_sample", False):
    sample_path = os.path.join(os.path.dirname(__file__), "test_data.csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state.use_sample = False
        st.session_state.data_source = f"uploaded: {uploaded_file.name}"
    else:
        df = pd.read_csv(sample_path)
        st.session_state.data_source = "sample: test_data.csv"

    feature_names = load_feature_names()
    missing_cols = [c for c in feature_names + [TARGET_COLUMN] if c not in df.columns]
    if missing_cols:
        st.error(f"CSV is missing required columns: {missing_cols}")
        st.stop()

    X = df[feature_names]
    y_true = df[TARGET_COLUMN]

    # Summary cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test Samples", len(df))
    c2.metric("Features Used", len(feature_names))
    c3.metric("Malignant (0)", int((y_true == 0).sum()))
    c4.metric("Benign (1)", int((y_true == 1).sum()))

    st.caption(f"Data source: {st.session_state.data_source}")

    # --- All models comparison ---
    st.markdown("### 📊 Model Comparison on Test Data")
    st.markdown(
        "I evaluated all 5 models on the **same test set** using 6 metrics "
        "(Accuracy, AUC, Precision, Recall, F1, MCC) as required in the assignment."
    )

    comparison_rows = []
    for model_name in MODEL_FILES:
        m = load_model(model_name)
        y_p = m.predict(X)
        y_pr = m.predict_proba(X)[:, 1]
        row = compute_metrics(y_true, y_p, y_pr)
        row["Model"] = model_name
        comparison_rows.append(row)

    comparison_df = pd.DataFrame(comparison_rows)[
        ["Model", "Accuracy", "AUC", "Precision", "Recall", "F1 Score", "MCC"]
    ]
    best_idx = comparison_df["Accuracy"].idxmax()
    best_model = comparison_df.loc[best_idx, "Model"]

    st.markdown(
        f'Best model on my test data: **{best_model}** '
        f'<span class="winner-badge">Accuracy {comparison_df.loc[best_idx, "Accuracy"]:.4f}</span>',
        unsafe_allow_html=True,
    )

    # Styled table — highlight best row
    def highlight_best(row):
        return [
            "background-color: #ecfdf5; font-weight: 600"
            if row.name == best_idx
            else ""
        ] * len(row)

    st.dataframe(
        comparison_df.style.apply(highlight_best, axis=1),
        use_container_width=True,
        hide_index=True,
    )

    # Bar chart comparing accuracy across all models
    fig_cmp, ax_cmp = plt.subplots(figsize=(8, 3.5))
    bar_colors = ["#2563eb" if m != best_model else "#059669" for m in comparison_df["Model"]]
    ax_cmp.bar(comparison_df["Model"], comparison_df["Accuracy"], color=bar_colors, edgecolor="white")
    ax_cmp.set_ylim(0.85, 1.02)
    ax_cmp.set_ylabel("Accuracy")
    ax_cmp.set_title("Accuracy Comparison — All 5 Models", fontweight="bold")
    ax_cmp.tick_params(axis="x", rotation=20)
    ax_cmp.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig_cmp)
    plt.close()

    st.markdown("---")

    # --- Selected model deep dive ---
    st.markdown(f"### 🔍 Deep Dive: {selected_model}")
    model = load_model(selected_model)
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    metrics = compute_metrics(y_true, y_pred, y_prob)

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Accuracy", metrics["Accuracy"])
    m2.metric("AUC", metrics["AUC"])
    m3.metric("Precision", metrics["Precision"])
    m4.metric("Recall", metrics["Recall"])
    m5.metric("F1 Score", metrics["F1 Score"])
    m6.metric("MCC", metrics["MCC"])

    tab1, tab2, tab3 = st.tabs(["Confusion Matrix", "ROC Curve", "Classification Report"])

    with tab1:
        fig_cm = plot_confusion_matrix(y_true, y_pred, selected_model)
        st.pyplot(fig_cm)
        plt.close()
        st.caption(
            "In medical diagnosis, false negatives (missing malignant cases) are especially costly. "
            "I checked recall carefully for each model."
        )

    with tab2:
        fig_roc = plot_roc_curve(y_true, y_prob, selected_model)
        st.pyplot(fig_roc)
        plt.close()

    with tab3:
        report_df = pd.DataFrame(
            classification_report(
                y_true, y_pred,
                target_names=["Malignant (0)", "Benign (1)"],
                output_dict=True,
            )
        ).transpose()
        st.dataframe(report_df.round(4), use_container_width=True)

    # Metric bar chart for selected model
    st.markdown("#### Metric breakdown")
    fig_bar = plot_metrics_bars(metrics, selected_model)
    st.pyplot(fig_bar)
    plt.close()

    with st.expander("Preview uploaded test data (first 10 rows)"):
        st.dataframe(df.head(10), use_container_width=True)

# Footer
st.markdown("---")
st.caption(
    f"ML Assignment 2 · {STUDENT_ID} · Breast Cancer Wisconsin (UCI) · "
    "Built with Streamlit + scikit-learn"
)
