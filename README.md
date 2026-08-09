# ML Assignment 2

**Student ID:** 2025AC05601  
**Course:** Machine Learning (M.Tech AIML/DSE)

---

## a. Problem Statement

Predict whether a breast tumor is **Malignant (0)** or **Benign (1)** using features from the Breast Cancer Wisconsin dataset. This is a binary classification problem. I trained 5 models on the same data, compared them using 6 metrics, and deployed the results in a Streamlit app.

---

## b. Dataset Description

| Property | Details |
|----------|---------|
| Source | [UCI Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)) |
| Samples | 569 |
| Features | 30 |
| Target | 0 = Malignant, 1 = Benign |

Train-test split: 75% train, 25% test. Test data is saved as `test_data.csv`.

---

## c. GitHub Repository Link

https://github.com/pratswinz/machine_learning_assignment_2

---

## d. Models Used

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|----|-----|
| Logistic Regression | 0.9860 | 0.9977 | 0.9889 | 0.9889 | 0.9889 | 0.9700 |
| Decision Tree | 0.9231 | 0.9234 | 0.9540 | 0.9222 | 0.9379 | 0.8378 |
| kNN | 0.9790 | 0.9845 | 0.9677 | 1.0000 | 0.9836 | 0.9555 |
| Naive Bayes | 0.9371 | 0.9878 | 0.9355 | 0.9667 | 0.9508 | 0.8644 |
| Random Forest | 0.9580 | 0.9950 | 0.9565 | 0.9778 | 0.9670 | 0.9098 |

### Observations on Model Performance

| ML Model Name | Observation about Model Performance |
|---------------|---------------------------------------|
| Logistic Regression | Best overall. Highest accuracy and AUC. Works well after scaling. |
| Decision Tree | Lowest accuracy. Easy to interpret but overfits on this data. |
| kNN | Good accuracy with recall 1.0 on test data. |
| Naive Bayes | Lower accuracy but decent AUC. |
| Random Forest | Better than single tree, but below Logistic Regression. |
| Overall best model | Logistic Regression |

---

## Streamlit App

**Live App Link:** (add after deployment on Streamlit Cloud)

### How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```
