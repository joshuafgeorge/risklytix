from flask import Flask, request, jsonify, render_template
import risklytic_final  # Import your risk analysis script

app = Flask(__name__)

# Route for rendering the HTML frontend (index.html)
@app.route('/')
def index():
    return render_template('index.html')

# Route for calculating the risk number
@app.route('/api/calculate_risk', methods=['POST'])
def calculate_risk():
    ticker_symbol = request.json.get('ticker')

    if not ticker_symbol:
        return jsonify({'error': 'Ticker symbol is required'}), 400
    
    try:
        # Run your analysis script
        risk_number = risklytic_final.main(ticker_symbol)
        
        # Convert risk_number to a standard float before returning it in JSON
        risk_number = float(risk_number)
        
        # Return the risk number in a JSON response
        return jsonify({'risk_number': risk_number})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
