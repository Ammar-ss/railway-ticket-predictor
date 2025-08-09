from flask import current_app as app, render_template, request, jsonify
import pandas as pd
from models.predictor import predict_confirmation

@app.route('/')
def index():
    """Renders the home page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles the prediction request from the front-end."""
    if request.method == 'POST':
        data = request.get_json()
        df = pd.DataFrame(data, index=[0])

        # Convert date strings to datetime objects
        df['Date of Journey'] = pd.to_datetime(df['Date of Journey'])
        df['Booking Date'] = pd.to_datetime(df['Booking Date'])

        prediction = predict_confirmation(df)
        
        return jsonify({'prediction': prediction})

