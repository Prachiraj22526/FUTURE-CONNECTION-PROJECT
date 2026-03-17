from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import os

app = Flask(__name__, template_folder="templates")

# -------------------------------
# Load required files
# -------------------------------
model = joblib.load("connection_type_xgb_model.joblib")
encoders = joblib.load("label_encoders.joblib")
model_features = joblib.load("model_features.joblib")

API_KEY = "mysecret123"

# -------------------------------
# Home Route
# -------------------------------
@app.route("/", methods=["GET"])
def home():
    return render_template(
        "index.html",
        tariff_values=list(encoders["TARIFF_ID"].classes_),
        source_values=list(encoders["APP_SOURCE"].classes_)
    )

# -------------------------------
# Prediction API
# -------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # API key validation
        if request.headers.get("x-api-key") != API_KEY:
            return jsonify({"error": "Invalid API Key"}), 401

        data = request.get_json()
        df = pd.DataFrame(data, index=[0])
        df = df[model_features]

        # Validate categorical values
        for col in ["TARIFF_ID", "APP_SOURCE"]:
            if df[col].iloc[0] not in encoders[col].classes_:
                return jsonify({
                    "error": f"Invalid value for {col}. Allowed values: {list(encoders[col].classes_)}"
                })

        # Encode
        df["TARIFF_ID"] = encoders["TARIFF_ID"].transform(df["TARIFF_ID"])
        df["APP_SOURCE"] = encoders["APP_SOURCE"].transform(df["APP_SOURCE"])

        # Predict
        pred = model.predict(df)
        result = encoders["CONN_TYPE"].inverse_transform(pred)

        return jsonify({"predicted_connection_type": result[0]})

    except Exception as e:
        return jsonify({"error": str(e)})

# -------------------------------
# Run App
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)
