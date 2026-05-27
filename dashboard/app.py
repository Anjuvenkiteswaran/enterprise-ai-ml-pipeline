import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/processed/customer_data.csv")

# Load churn model
model = joblib.load("models/churn_model.pkl")

st.title("Enterprise AI Customer Analytics Dashboard")

st.subheader("Customer Dataset Overview")

st.dataframe(df.head())

# KPI Metrics
total_customers = len(df)
avg_usage = round(df["feature_usage_score"].mean(), 2)
avg_sentiment = round(df["negative_sentiment_ratio"].mean(), 2)

col1, col2, col3 = st.columns(3)

col1.metric("Total Customers", total_customers)
col2.metric("Avg Feature Usage", avg_usage)
col3.metric("Avg Negative Sentiment", avg_sentiment)

# Churn Distribution
st.subheader("Churn Distribution")

fig, ax = plt.subplots()

df["churn"].value_counts().plot(
    kind="bar",
    ax=ax
)

st.pyplot(fig)

# Customer Prediction Section
st.subheader("Predict Customer Churn")

tenure = st.slider("Tenure Months", 1, 60, 12)
active_days = st.slider("Monthly Active Days", 1, 30, 10)
session_duration = st.slider(
    "Average Session Duration",
    1.0,
    120.0,
    30.0
)

support_tickets = st.slider(
    "Support Ticket Count",
    0,
    20,
    2
)

payment_delay = st.slider(
    "Payment Delay Count",
    0,
    10,
    1
)

feature_usage = st.slider(
    "Feature Usage Score",
    0.0,
    100.0,
    50.0
)

subscription_plan = st.selectbox(
    "Subscription Plan",
    ["Basic", "Pro", "Enterprise"]
)

negative_sentiment = st.slider(
    "Negative Sentiment Ratio",
    0.0,
    1.0,
    0.5
)

plan_mapping = {
    "Basic": 0,
    "Enterprise": 1,
    "Pro": 2
}

input_df = pd.DataFrame({
    "tenure_months": [tenure],
    "monthly_active_days": [active_days],
    "avg_session_duration": [session_duration],
    "support_ticket_count": [support_tickets],
    "payment_delay_count": [payment_delay],
    "feature_usage_score": [feature_usage],
    "subscription_plan": [plan_mapping[subscription_plan]],
    "negative_sentiment_ratio": [negative_sentiment]
})

if st.button("Predict Churn"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(
            f"High Churn Risk | Probability: {probability:.2f}"
        )
    else:
        st.success(
            f"Low Churn Risk | Probability: {probability:.2f}"
        )