# ML Assignment 2 - Breast Cancer Classification

**Student ID:** 2025AC05601  
**Course:** Machine Learning (M.Tech AIML/DSE)  
**Dataset:** Breast Cancer Wisconsin (Diagnostic) - UCI ML Repository

---

## a. Problem Statement

The goal of this assignment is to build and compare multiple classification models that can predict whether a breast tumor is **malignant (cancerous)** or **benign (non-cancerous)** based on features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass.

This is a **binary classification** problem. I implemented 5 different ML models on the same dataset, evaluated them using 6 metrics, and deployed the results on a Streamlit web application.

---

## b. Dataset Description

| Property | Details |
|----------|---------|
| **Source** | [UCI ML Repository - Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)) |
| **Samples** | 569 instances |
| **Features** | 30 (computed from cell nuclei characteristics) |
| **Target** | 0 = Malignant, 1 = Benign |
| **Class Distribution** | Benign: 357, Malignant: 212 |

**Feature categories:**
- Mean, standard error, and worst value for: radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension

The dataset meets the assignment requirements (minimum 12 features and 500 instances). I used a 75-25 train-test split and saved the test portion as `test_data.csv`.

---

## c. GitHub Repository Link

**Repository:** https://github.com/pratswinz/machine_learning_assignment_2

**Repository structure:**
```
Machine_learning_assignment_2/
├── app.py                  # Streamlit web application
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── test_data.csv           # Test dataset (25% holdout)
└── model/
    ├── train_models.py     # Training script
    ├── 2025AC05601_ml_assignment2.ipynb
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── random_forest.pkl
    ├── feature_names.json
    └── metrics.json
```

---

## d. Models Used

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|----|-----|
| Logistic Regression | 0.9860 | 0.9977 | 0.9889 | 0.9889 | 0.9889 | 0.9700 |
| Decision Tree | 0.9231 | 0.9234 | 0.9540 | 0.9222 | 0.9379 | 0.8378 |
| kNN | 0.9790 | 0.9845 | 0.9677 | 1.0000 | 0.9836 | 0.9555 |
| Naive Bayes | 0.9371 | 0.9878 | 0.9355 | 0.9667 | 0.9508 | 0.8644 |
| Random Forest (Ensemble) | 0.9580 | 0.9950 | 0.9565 | 0.9778 | 0.9670 | 0.9098 |

### Observations on Model Performance

| ML Model Name | Observation about Model Performance |
|---------------|---------------------------------------|
| Logistic Regression | Best overall model on this dataset. Achieved the highest accuracy (98.6%), AUC (0.9977), and F1 score (0.9889). Works well because the classes are largely linearly separable after scaling. |
| Decision Tree | Lowest accuracy among all models (92.3%). Single decision trees tend to overfit and create overly complex boundaries. Good interpretability but weaker generalization on this dataset. |
| kNN | Strong performance with 97.9% accuracy. Perfect recall (1.0) — it did not miss any malignant cases, which is important in medical diagnosis. Slightly lower precision compared to Logistic Regression. |
| Naive Bayes | Moderate accuracy (93.7%) but surprisingly high AUC (0.9878). The Gaussian assumption does not fully match the feature distributions, but the model still captures the overall class separation reasonably well. |
| Random Forest (Ensemble) | Solid ensemble performance with 95.8% accuracy and 0.995 AUC. Better than a single Decision Tree as expected, but did not surpass Logistic Regression on this dataset. |
| **Overall best model** | **Logistic Regression** — highest accuracy, AUC, precision, recall, F1, and MCC on my test split. |

---

## Streamlit App

**Live App Link:** [Add your Streamlit Cloud link after deployment]

```
https://YOUR_APP_NAME.streamlit.app
```

### App Features
- CSV file upload for test data
- Model selection dropdown (5 models)
- Display of all 6 evaluation metrics
- Confusion matrix visualization
- Classification report table

### How to Run Locally
```bash
pip install -r requirements.txt
python model/train_models.py    # Train models (first time only)
streamlit run app.py
```

---

## References

- UCI ML Repository: Breast Cancer Wisconsin (Diagnostic) Dataset
- Scikit-learn documentation for classification metrics
- Streamlit documentation for deployment

---

*Submitted as part of ML Assignment 2, BITS WILP — Deadline: 18-Aug-2026*
