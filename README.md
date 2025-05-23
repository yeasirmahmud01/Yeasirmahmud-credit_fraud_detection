# 💳 Credit Card Fraud Detection Using Machine Learning

This project focuses on detecting fraudulent credit card transactions using supervised machine learning. It includes preprocessing, model training, evaluation, and deployment via a Streamlit-based web app.

---

## 📁 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Streamlit App](#streamlit-app)
- [How to Run](#how-to-run)
- [Key Learnings & Challenges](#key-learnings--challenges)
- [Credits](#credits)

---

## 📌 Overview

Credit card fraud causes billions in global losses. Detecting it automatically is crucial yet difficult due to its rarity. This project builds and deploys a Random Forest-based fraud detection system using real anonymized transaction data.

---

## 📊 Dataset

- **Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size**: 284,807 transactions (492 frauds ≈ 0.172%)
- **Features**:
  - `Time`: Seconds since first transaction
  - `Amount`: Transaction amount
  - `V1`–`V28`: PCA-anonymized features
  - `Class`: Target (0 = normal, 1 = fraud)

---

## 🧠 Methodology

### Preprocessing
- Removed null values
- Scaled `Time` and `Amount` using `StandardScaler`
- 80/20 train-test split with stratified sampling

### Models Evaluated
- Logistic Regression
- Decision Tree
- Naive Bayes
- K-Nearest Neighbors (KNN)
- **Random Forest (Best Performance)**

### Metrics Used
- Precision, Recall, F1 Score
- ROC-AUC

---

## 📈 Results

| Model              | Accuracy | Precision | Recall | AUC   |
|-------------------|----------|-----------|--------|-------|
| Logistic Regression | 0.9986 | 0.86      | 0.62   | 0.951 |
| Decision Tree       | 0.9990 | 0.89      | 0.70   | 0.974 |
| Naive Bayes         | 0.9630 | 0.13      | 0.81   | 0.831 |
| KNN (k=5)           | 0.9985 | 0.86      | 0.59   | 0.938 |
| **Random Forest**   | **0.9992** | **0.91** | **0.76** | **0.980** |

---

## 🌐 Streamlit App

A user-friendly frontend built with Streamlit for fraud detection.

### 🔧 Features:
- Upload CSV with required columns (`Time`, `V1`–`V28`, `Amount`)
- Validates input format and structure
- Shows fraud probabilities and prediction summary
- Interactive bar chart of fraud vs. normal transactions
- Downloadable results as CSV
- Custom CSS for modern UI
- Footer with contact & GitHub info

---

## ▶️ How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Launch app
streamlit run credit_card_fraud_app.py
