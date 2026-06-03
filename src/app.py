from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load artifacts
model = joblib.load("models/logistic_regression_model.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_names = joblib.load("models/feature_names.pkl")

numerical_cols = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]


@app.route("/")
def home():
    return "Customer Churn Prediction API Running"


@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    df = pd.DataFrame([data])

    df = pd.get_dummies(
        df,
        drop_first=False
    )

    df = df.reindex(
        columns=feature_names,
        fill_value=0
    )

    df[numerical_cols] = scaler.transform(
        df[numerical_cols]
    )

    prediction = int(model.predict(df)[0])

    churn_probability = float(
        model.predict_proba(df)[0][1]
    )

    return jsonify(
        {
            "prediction": prediction,
            "churn_probability": churn_probability
        }
    )


if __name__ == "__main__":
    app.run(debug=True)