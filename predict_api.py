from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)  # allows your Next.js app to call this API

# ── Load model and encoders once at startup ───────────────────────────────
print("Loading model...")
model    = joblib.load('cafepay_model.pkl')
encoders = joblib.load('encoder.pkl')
features = joblib.load('features.pkl')
print("Model loaded. API is ready.")

# ── Health check endpoint ─────────────────────────────────────────────────
@app.route('/', methods=['GET'])
def health():
    return jsonify({'status': 'CafePay ML API is running'})

# ── Main prediction endpoint ──────────────────────────────────────────────
@app.route('/predict', methods=['POST'])
def predict():
    try:
        tx = request.json

        # Validate required fields
        required = [
            'transaction_amount', 'category', 'vendor_category',
            'daily_tx_count', 'wallet_balance_pct',
            'monthly_budget_used_pct', 'category_budget_used_pct',
            'days_elapsed_in_month', 'spend_velocity',
            'days_remaining_budget', 'is_weekend', 'hour_of_day'
        ]
        for field in required:
            if field not in tx:
                return jsonify({'error': f'Missing field: {field}'}), 400

        # Encode text fields
        try:
            category_enc = encoders['category'].transform(
                [tx['category']])[0]
        except ValueError:
            category_enc = 0  # unknown category defaults to 0

        try:
            vendor_enc = encoders['vendor_category'].transform(
                [tx['vendor_category']])[0]
        except ValueError:
            vendor_enc = 0

        # Build the feature row
        row = [[
            float(tx['transaction_amount']),
            int(category_enc),
            int(vendor_enc),
            int(tx['daily_tx_count']),
            float(tx['wallet_balance_pct']),
            float(tx['monthly_budget_used_pct']),
            float(tx['category_budget_used_pct']),
            int(tx['days_elapsed_in_month']),
            float(tx['spend_velocity']),
            float(tx['days_remaining_budget']),
            int(tx['is_weekend']),
            int(tx['hour_of_day'])
        ]]

        # Run prediction
        prediction = model.predict(row)[0]

        return jsonify({
            'expenditure_level': str(prediction[0]),
            'duration_risk':     str(prediction[1]),
            'behaviour_class':   str(prediction[2]),
            'status':            'success'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)