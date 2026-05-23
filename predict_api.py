from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os

app = Flask(__name__)
CORS(app)

model    = joblib.load('cafepay_model.pkl')
encoders = joblib.load('encoder.pkl')

@app.route('/', methods=['GET'])
def health():
    return jsonify({'status': 'CafePay ML API is running'})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        tx = request.json

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

        try:
            category_enc = encoders['category'].transform(
                [tx['category']])[0]
        except ValueError:
            category_enc = 0

        try:
            vendor_enc = encoders['vendor_category'].transform(
                [tx['vendor_category']])[0]
        except ValueError:
            vendor_enc = 0

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
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port)