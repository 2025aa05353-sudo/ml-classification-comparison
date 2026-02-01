# 🍷🥂 Wine Quality Classifications for (Red & White Wine) using Machine Learning

---

## a. Problem Statement

The objective of this project is to build and compare multiple machine learning classification models to predict the **quality of wine** based on its physicochemical properties.

The wine quality prediction problem is formulated as a **multiclass classification task**, where wine samples are classified into quality scores ranging from **3 to 9**.

The project aims to:
- Train different machine learning models on the wine dataset  
- Evaluate their performance using standard classification metrics  
- Compare the models to identify the most suitable one for wine quality prediction  

---

## b. Dataset Description

**Dataset Source:**  
[Kaggle (UCI Machine Learning Repository)](https://www.kaggle.com/datasets/ruthgn/wine-quality-data-set-red-white-wine)

**Dataset Type:**  
Red and White Wine (combined)

**Number of Instances:**  
- Total: 6,497  
- Red wine: 1,599  
- White wine: 4,898  

**Number of Features:**  
- 12 total  
  - 1 categorical feature (`type`: red / white)  
  - 11 physicochemical features  

### Feature Description

| Feature | Description |
|-------|-------------|
| Type | Red or White wine |
| Fixed Acidity | Non-volatile acids |
| Volatile Acidity | Acetic acid amount |
| Citric Acid | Adds freshness |
| Residual Sugar | Sugar after fermentation |
| Chlorides | Salt content |
| Free Sulfur Dioxide | Prevents microbial growth |
| Total Sulfur Dioxide | Total SO₂ concentration |
| Density | Wine density |
| pH | Acidity level |
| Sulphates | Wine preservative |
| Alcohol | Alcohol content |
| Quality | Target variable |

**Target Variable:** `quality`  
**Target Classes:** 7 (Quality scores: 3, 4, 5, 6, 7, 8, 9)  
**Problem Type:** Multiclass Classification  

---

## c. Models Used and Performance Comparison

The following six machine learning models were trained and evaluated:

- Logistic Regression  
- Decision Tree  
- k-Nearest Neighbors (kNN)  
- Naive Bayes  
- Random Forest (Ensemble)  
- XGBoost (Ensemble)  

### Evaluation Metrics Used
- Accuracy  
- AUC (One-vs-Rest)  
- Precision (Macro-average)  
- Recall (Macro-average)  
- F1-Score (Macro-average)  
- Matthews Correlation Coefficient (MCC)  

### Model Comparison Table

| ML Model Name | Accuracy | Precision | Recall | F1 | MCC | AUC |
|--------------|----------|-----------|--------|----|-----|-----|
| Logistic Regression | 0.5385 | 0.2977 | 0.2243 | 0.2261 | 0.2676 | 0.7823 |
| Decision Tree | 0.6023 | 0.3283 | 0.3324 | 0.3301 | 0.4075 | 0.6241 |
| kNN | 0.5369 | 0.3199 | 0.2577 | 0.2667 | 0.2902 | 0.6840 |
| Naive Bayes | 0.3515 | 0.2344 | 0.3425 | 0.2030 | 0.1250 | 0.6901 |
| Random Forest (Ensemble) | 0.6146 | 0.4168 | 0.2643 | 0.2710 | 0.3933 | 0.7982 |
| XGBoost (Ensemble) | 0.6454 | 0.4379 | 0.3367 | 0.3607 | 0.4534 | 0.8342 |

---

## d. Observations on the Performance of Each Model

| ML Model Name | Observation about Model Performance |
|--------------|-------------------------------------|
| Logistic Regression | Shows moderate accuracy but low recall and F1-score, indicating difficulty in capturing complex non-linear relationships present in the data. |
| Decision Tree | Performs better than linear models with improved recall and MCC, but lower AUC suggests limited generalization and potential overfitting. |
| kNN | Achieves comparable accuracy to Logistic Regression but lower recall, showing sensitivity to class distribution and feature scaling. |
| Naive Bayes | Exhibits the lowest accuracy and F1-score due to the strong assumption of feature independence, which does not hold well for this dataset. |
| Random Forest (Ensemble) | Demonstrates strong and stable performance with higher accuracy, precision, and AUC, benefiting from ensemble learning and reduced overfitting. |
| XGBoost (Ensemble) | Achieves the best overall performance across most metrics, including highest accuracy, AUC, F1-score, and MCC, indicating superior learning of complex patterns. |

---

## Summary Insight:

- Ensemble models outperform individual models.  
- **XGBoost** is the most effective model for wine quality classification.  
- Lower-performing models struggle mainly due to linear assumptions or oversimplified probability assumptions.
