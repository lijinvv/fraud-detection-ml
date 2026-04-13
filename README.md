# Credit Card Fraud Detection

## Overview
End-to-end machine learning system detecting fraudulent credit card transactions from 284,807 real transactions (0.17% fraud rate).

## The Challenge
Severely imbalanced dataset — 99.83% legitimate, 0.17% fraud. Evaluation focused on ROC-AUC, Precision, Recall and F1 score.

## Approach
1. Exploratory data analysis and class imbalance visualisation
2. Feature scaling using StandardScaler
3. SMOTE oversampling (227,451 samples per class after resampling)
4. Three models trained and compared
5. Feature importance analysis

## Results

| Model | ROC-AUC |
|---|---|
| Logistic Regression | 0.9698 |
| Random Forest | 0.9688 |
| XGBoost | 0.9792 |

Best model: XGBoost — ROC-AUC 0.9792

## Confusion Matrix (Random Forest)
- 56,847 legitimate transactions correctly identified
- 80 of 98 fraudulent transactions caught (82% recall)
- Only 17 false positives from 56,864 legitimate transactions

## Key Findings
- V14 was the dominant fraud predictor (importance score 0.22)
- V10 and V4 were second and third strongest signals
- SMOTE significantly improved minority class recall
- XGBoost outperformed Random Forest and Logistic Regression

## Dataset
Download creditcard.csv from Kaggle:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
Place it in the root folder before running the notebook.

## Tech Stack
Python, Pandas, NumPy, Scikit-Learn, XGBoost, Imbalanced-Learn, Matplotlib, Seaborn

## How to Run
pip install -r requirements.txt
jupyter notebook fraud_detection.ipynb
