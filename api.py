from flask import Flask, request, jsonify
from flask_cors import CORS
from predict import CreditRiskModel
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize model
try:
    model = CreditRiskModel()
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/api/defaults', methods=['GET'])
def get_defaults():
    if not model:
        return jsonify({"error": "Model not loaded"}), 500
    return jsonify(model.ideal_defaults())

@app.route('/api/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.json
    if not data:
        return jsonify({"error": "No input data provided"}), 400

    try:
        # Predict
        proba, pred = model.predict_proba(data)
        label, color = CreditRiskModel.risk_category(proba)

        # SHAP explanation
        shap_df, meta = model.explain_shap(data)

        # Convert SHAP DataFrame to list of dicts for JSON response
        shap_records = shap_df.head(20).to_dict(orient='records')

        response = {
            "probability": proba,
            "prediction": pred,
            "risk_label": label,
            "risk_color": color,
            "shap_values": shap_records,
            "base_value": meta['base_value'],
            "prediction_probability": meta['prediction_probability']
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
