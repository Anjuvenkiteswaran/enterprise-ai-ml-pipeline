# Enterprise AI ML Pipeline

End-to-end machine learning project for enterprise customer analytics, built to demonstrate predictive analytics, feature engineering, model evaluation, and production-oriented ML workflows.

## Project Overview

This project simulates an enterprise SaaS/customer platform where the business wants to identify customers at risk of churn and understand the key drivers behind customer behavior.

The pipeline includes:

- Synthetic enterprise customer dataset generation
- Churn prediction model
- Feature engineering
- Class imbalance analysis
- SMOTE-based balancing
- ROC-AUC and classification evaluation
- Model serialization for production use

## Business Problem

Enterprise customers interact with multiple product features, submit support tickets, delay payments, and show different engagement patterns. The goal is to predict which customers are likely to churn so the product and customer success teams can take timely action.

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Imbalanced-learn
- SMOTE
- Random Forest
- Jupyter Notebook
- Joblib
- Git & GitHub

## Dataset

A synthetic enterprise customer dataset was created with 5,000 customer records.

Key features include:

- Customer tenure
- Monthly active days
- Average session duration
- Support ticket count
- Payment delay count
- Feature usage score
- Subscription plan
- Negative sentiment ratio
- Churn flag

## Project Structure

```
enterprise-ai-ml-pipeline/
│
├── data/
│   └── processed/
│       └── customer_data.csv
│
├── notebooks/
│   ├── 01_generate_dataset.ipynb
│   └── 02_churn_prediction.ipynb
│
├── models/
│   └── churn_model.pkl
│
├── README.md
├── requirements.txt
└── .gitignore
