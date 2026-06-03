from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib

app = Flask(__name__)

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
    return render_template("index.html")


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

    probability = float(
        model.predict_proba(df)[0][1]
    )

    return jsonify({
        "prediction": prediction,
        "churn_probability": probability
    })


@app.route("/predict_form", methods=["POST"])
def predict_form():

    data = {
        "gender": request.form["gender"],
        "SeniorCitizen": 0,
        "Partner": request.form["Partner"],
        "Dependents": request.form["Dependents"],
        "tenure": int(request.form["tenure"]),
        "PhoneService": request.form["PhoneService"],
        "MultipleLines": request.form["MultipleLines"],
        "InternetService": request.form["InternetService"],
        "OnlineSecurity": request.form["OnlineSecurity"],
        "OnlineBackup": request.form["OnlineBackup"],
        "DeviceProtection": request.form["DeviceProtection"],
        "TechSupport": request.form["TechSupport"],
        "StreamingTV": request.form["StreamingTV"],
        "StreamingMovies": request.form["StreamingMovies"],
        "Contract": request.form["Contract"],
        "PaperlessBilling": request.form["PaperlessBilling"],
        "PaymentMethod": request.form["PaymentMethod"],
        "MonthlyCharges": float(request.form["MonthlyCharges"]),
        "TotalCharges": float(request.form["TotalCharges"])
    }

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

    probability = float(
        model.predict_proba(df)[0][1]
    )

    return f"""
    <h1>Prediction Result</h1>

    <h2>
    Churn Prediction: {prediction}
    </h2>

    <h2>
    Churn Probability: {probability:.4f}
    </h2>

    <br>

    <a href="/">
        Go Back
    </a>
    """


if __name__ == "__main__":
    app.run(debug=True)