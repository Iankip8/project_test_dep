from flask import Flask, request, jsonify
import joblib
import os

# Construct the relative path to the model file
current_directory = os.path.dirname(__file__)
model_path = os.path.join(current_directory, 'voting_classifier_pipeline2.pkl')

# Load the trained model
model = joblib.load(model_path)

app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to the model API!"

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the POST request
    data = request.get_json()
    
    # Extract input data
    input_data = data['input']  # Assuming the input data is sent as JSON
    
    # Make prediction
    prediction = model.predict([input_data])
    
    # Return the result as JSON
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    # Get the port from the environment variable, or default to 5000
    port = int(os.environ.get('PORT', 5000))
    # Run the app, specifying host='0.0.0.0' to accept connections from any IP address
    app.run(host='0.0.0.0', port=port, debug=True)
