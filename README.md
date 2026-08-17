# ML Assignment 2

**Student ID:** 2025AC05601  
**Course:** Machine Learning (M.Tech AIML/DSE)

---

## a. Problem Statement

Breast cancer diagnosis often uses a Fine Needle Aspiration (FNA) test. From the cell sample, 30 numerical features are measured (radius, texture, perimeter, area, smoothness, and related mean / error / worst values). The task is to classify each sample as **Malignant (0)** or **Benign (1)**.

This is a binary classification problem. Missing a malignant tumour (false negative for cancer) and raising a false alarm (false positive) are both costly, so Accuracy alone is not enough. All five models are compared using Accuracy, AUC, Precision, Recall, F1 Score and MCC on the same test set.

Models used:

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbor (kNN)  
4. Naive Bayes (Gaussian)  
5. Random Forest (Ensemble)

The trained models are deployed in a Streamlit app. Only test CSV data is uploaded. The app shows metrics, confusion matrix, classification report, and a comparison of all five models.

---

## b. Dataset Description

| Property | Details |
|----------|---------|
| Dataset | Breast Cancer Wisconsin (Diagnostic) |
| Source | [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)) |
| Samples | 569 (greater than 500 required) |
| Features | 30 real-valued cell-nucleus measurements (greater than 12 required) |
| Problem type | Binary classification |
| Target | `target` — 0 = Malignant, 1 = Benign |
| Class mix (full data) | 212 Malignant, 357 Benign |
| Missing values | None |
| Loaded using | `sklearn.datasets.load_breast_cancer()` |

Train-test split: 75% train (426 samples), 25% test (143 samples), stratified, `random_state=42`. Stratified split keeps the malignant/benign ratio similar in both sets. The held-out test set has **53 malignant** and **90 benign** rows. This test portion is saved as `test_data.csv` for the Streamlit app.

**Preprocessing**

1. Stratified train-test split (`test_size=0.25`, `random_state=42`) so class ratios are preserved.  
2. `StandardScaler` is used **inside a Pipeline** for Logistic Regression, kNN and Naive Bayes. The scaler is fit on train only, then applied to test, so there is no leakage.  
3. Decision Tree and Random Forest are trained without scaling. Tree splits do not depend on feature magnitude.  
4. All five models see the same 426 training rows and the same 143 test rows, so the comparison is fair.

---

## c. GitHub Repository Link

https://github.com/pratswinz/machine_learning_assignment_2

The repository contains `app.py`, `requirements.txt`, `README.md`, `test_data.csv`, and the `model/` folder with saved `.pkl` files, `feature_names.json`, `metrics.json`, `train_models.py` and the Jupyter notebook.

---

## d. Models Used

All five models are trained on the same training set and evaluated on the same 143-row test set. Precision, Recall and F1 are reported for the positive class **Benign (1)**, which is sklearn’s default and the same setting used in the Streamlit app. Accuracy, AUC and MCC use both classes.

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|----|-----|
| Logistic Regression | 0.9860 | 0.9977 | 0.9889 | 0.9889 | 0.9889 | 0.9700 |
| Decision Tree | 0.9231 | 0.9234 | 0.9540 | 0.9222 | 0.9379 | 0.8378 |
| kNN | 0.9790 | 0.9845 | 0.9677 | 1.0000 | 0.9836 | 0.9555 |
| Naive Bayes | 0.9371 | 0.9878 | 0.9355 | 0.9667 | 0.9508 | 0.8644 |
| Random Forest (Ensemble) | 0.9580 | 0.9950 | 0.9565 | 0.9778 | 0.9670 | 0.9098 |

### Settings used

| Model | Settings |
|-------|----------|
| Logistic Regression | `StandardScaler` + `max_iter=1000`, `random_state=42` |
| Decision Tree | default tree (`max_depth=None`), `random_state=42` |
| kNN | `StandardScaler` + `n_neighbors=5` |
| Naive Bayes | `StandardScaler` + GaussianNB |
| Random Forest (Ensemble) | `n_estimators=100`, `random_state=42` |

### Observations on Model Performance

| ML Model Name | Observation about Model Performance |
|---------------|---------------------------------------|
| Logistic Regression | Strongest model on this test set. Highest Accuracy (0.9860), AUC (0.9977), Precision, F1 and MCC. Confusion matrix is [[52, 1], [1, 89]] — only 2 mistakes in 143 samples. After StandardScaler, the 30 features are on a comparable scale and a linear boundary separates the two classes well. AUC 0.9977 also shows that the predicted probabilities rank malignant vs benign samples almost perfectly. |
| Decision Tree | Weakest model. Lowest Accuracy (0.9231) and lowest AUC (0.9234). Confusion matrix is [[49, 4], [7, 83]] — 11 errors. A single unpruned tree (`max_depth=None`) can fit training noise. Trees also output hard 0/1-style probabilities, which lowers AUC even when accuracy is still usable. The tree is easy to interpret, but it overfits this 30-feature dataset. |
| kNN | Second-best Accuracy (0.9790). Recall for Benign is 1.0000: all 90 benign test rows are found (0 missed). Confusion matrix is [[50, 3], [0, 90]]. The 3 errors are malignant samples predicted as benign, so Recall=1.0 does not mean cancer was never missed. kNN is distance-based, so StandardScaler is required. `n_neighbors=5` worked well on this split. |
| Naive Bayes | Mid-pack Accuracy (0.9371) but still a high AUC (0.9878). Confusion matrix is [[47, 6], [3, 87]]. GaussianNB assumes features are independent given the class. In this dataset radius, perimeter and area describe the same nucleus size, so that assumption is weak and extra false positives appear. Ranking quality (AUC) stays decent even though the hard labels are weaker than Logistic Regression. |
| Random Forest (Ensemble) | Clearly better than one Decision Tree: Accuracy 0.9580 vs 0.9231, MCC 0.9098 vs 0.8378, AUC 0.9950 vs 0.9234. Averaging 100 trees reduces the overfitting seen in the single tree. Confusion matrix is [[49, 4], [2, 88]] (6 errors). On this dataset it still sits below Logistic Regression on Accuracy, Precision, F1 and MCC, which matches a problem that is already close to linearly separable. |
| Overall Winner for your dataset? | **Logistic Regression.** It wins 5 of 6 metrics. MCC 0.9700 is a strong result for a 2-class medical task, and AUC 0.9977 shows well-ranked probabilities. kNN wins only Recall (1.0000) for the Benign class. For this data, the scaled linear model is the most reliable overall. |

### Metric-wise Best Models

| Metric | Best Model | Value |
|--------|------------|-------|
| Accuracy | Logistic Regression | 0.9860 |
| AUC | Logistic Regression | 0.9977 |
| Precision | Logistic Regression | 0.9889 |
| Recall | kNN | 1.0000 |
| F1 Score | Logistic Regression | 0.9889 |
| MCC | Logistic Regression | 0.9700 |

---

## Streamlit App

**Live App Link:** https://machinelearningassignment2git-8olbkbqtdsnsd9xb4ykchy.streamlit.app/

The app includes:

- CSV upload for test data (`test_data.csv`)
- Model selection dropdown (all 5 models)
- Evaluation metrics (Accuracy, AUC, Precision, Recall, F1, MCC)
- Confusion matrix and classification report
- Comparison table of all 5 models on the uploaded test data

### How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```
