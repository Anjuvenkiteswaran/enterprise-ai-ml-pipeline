# Enterprise AI Customer Analytics Platform

An end-to-end enterprise machine learning platform designed for predictive analytics, recommendation systems, NLP sentiment analysis, and production-style ML deployment.

This project simulates a real-world enterprise AI system where customer engagement, product usage, support interactions, and sentiment are analyzed to predict churn risk and generate actionable business insights.

---

# GitHub Repository

GitHub: https://github.com/Anjuvenkiteswaran/enterprise-ai-ml-pipeline

---

# Project Highlights

- Enterprise churn prediction pipeline
- Recommendation system for next-best product/module
- NLP sentiment analysis for support tickets
- FastAPI production deployment
- Interactive Streamlit analytics dashboard
- SQL-based feature engineering layer
- MLflow experiment tracking
- PySpark feature engineering pipeline
- Production-style ML workflow and model persistence

---

# Business Problem

Enterprise SaaS platforms often struggle with:

- Customer churn
- Poor customer engagement
- Negative support experiences
- Low feature adoption
- Product recommendation optimization

This platform addresses these problems through predictive analytics and machine learning workflows.

---

# Core Components

## 1. Churn Prediction System

Built a supervised machine learning pipeline to predict customer churn risk using:

- Customer engagement
- Session activity
- Support ticket frequency
- Payment delays
- Feature usage
- Sentiment indicators

### Techniques Used

- Random Forest Classifier
- SMOTE imbalance handling
- ROC-AUC evaluation
- Feature importance analysis

### ML Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

---

## 2. Recommendation Engine

Built a recommendation system to suggest next-best enterprise modules/products using customer usage similarity.

### Techniques Used

- Cosine Similarity
- Collaborative Filtering Logic
- User-Module Interaction Matrix

### Example Recommendation

```text
Customer: CUST_10

Recommended Modules:
- AI Forecasting
- Fraud Detection
- Workflow Optimizer
