from flask import Flask, request, jsonify
from flask_cors import CORS  # Allows requests from your React frontend
import risklytic_final  # Your risk analysis script

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend to communicate with Flask backend

# API route for calculating the risk number
@app.route('/api/calculate_risk', methods=['POST'])
def calculate_risk():
    ticker_symbol = request.json.get('ticker')

    if not ticker_symbol:
        return jsonify({'error': 'Ticker symbol is required'}), 400

    try:
        # Run your analysis script
        risk_number = risklytic_final.main(ticker_symbol)

        # Convert risk_number to a float
        risk_number = float(risk_number)

        return jsonify({'risk_number': risk_number})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
