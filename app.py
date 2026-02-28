from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)  # 🔥 This fixes the frontend connection issue

model = joblib.load("loan_pipeline.pkl")

@app.route("/")
def home():
    return "Loan API Running 🚀"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    input_df = pd.DataFrame([data])

    prediction = model.predict(input_df)[0]

    # Force probability safe handling
    try:
        probability = model.predict_proba(input_df)[0][1]
        probability = round(float(probability) * 100, 2)
    except:
        probability = 0.0

    return jsonify({
        "prediction": int(prediction),
        "default_probability": probability
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)