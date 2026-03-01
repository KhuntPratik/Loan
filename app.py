from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

model = joblib.load("random_forest_pipeline.pkl")

@app.route("/")
def home():
    return "Loan API Running 🚀"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    input_df = pd.DataFrame([data])

    prediction = model.predict(input_df)[0]

    # Handle probability safely
    try:
        probability = model.predict_proba(input_df)[0][1]
        probability = round(float(probability) * 100, 2)
    except:
        probability = 0.0

    # 🔥 Convert prediction to approval message
    if prediction == 1:
        loan_status = "Loan Rejected ❌"
    else:
        loan_status = "Loan Approved ✅"

    return jsonify({
        "loan_status": loan_status,
        "default_probability_percent": probability
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))