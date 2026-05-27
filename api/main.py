from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(
    title="Enterprise AI ML Pipeline API",
    description="API for churn prediction and NLP sentiment analysis",
    version="1.0"
)

churn_model = joblib.load("models/churn_model.pkl")
nlp_model = joblib.load("models/nlp_sentiment_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")


class ChurnInput(BaseModel):
    tenure_months: int
    monthly_active_days: int
    avg_session_duration: float
    support_ticket_count: int
    payment_delay_count: int
    feature_usage_score: float
    subscription_plan: int
    negative_sentiment_ratio: float


class SentimentInput(BaseModel):
    ticket_text: str


@app.get("/")
def home():
    return {"message": "Enterprise AI ML Pipeline API is running"}


@app.post("/predict-churn")
def predict_churn(data: ChurnInput):
    input_df = pd.DataFrame([data.dict()])
    prediction = churn_model.predict(input_df)[0]
    probability = churn_model.predict_proba(input_df)[0][1]

    return {
        "churn_prediction": int(prediction),
        "churn_probability": round(float(probability), 4)
    }


@app.post("/analyze-sentiment")
def analyze_sentiment(data: SentimentInput):
    text_vectorized = vectorizer.transform([data.ticket_text])
    prediction = nlp_model.predict(text_vectorized)[0]

    return {
        "ticket_text": data.ticket_text,
        "predicted_sentiment": prediction
    }