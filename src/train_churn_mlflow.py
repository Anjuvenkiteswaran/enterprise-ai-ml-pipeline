import pandas as pd
import mlflow
import mlflow.sklearn
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


df = pd.read_csv("data/processed/customer_data.csv")

encoder = LabelEncoder()
df["subscription_plan"] = encoder.fit_transform(df["subscription_plan"])

X = df.drop(columns=["customer_id", "churn"])
y = df["churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    random_state=42,
    class_weight="balanced"
)

mlflow.set_experiment("enterprise_churn_prediction")

with mlflow.start_run():
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("max_depth", 8)
    mlflow.log_param("class_weight", "balanced")

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", auc)

    mlflow.sklearn.log_model(model, "churn_model")

    joblib.dump(model, "models/churn_model_mlflow.pkl")

    print("Accuracy:", accuracy)
    print("F1 Score:", f1)
    print("ROC-AUC:", auc)
    print("Model saved and logged in MLflow")