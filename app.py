from flask import Flask, request, jsonify
import joblib

# Load the trained model
model_path = "C:\\Users\\Ian\\Documents\\fraiton\\Phase4Project\\NLP-tweet-sentiment-project\\voting_classifier_pipeline2.pkl"
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
    
    # Convert input data to the appropriate format
    # Example: input_data = [input_data] if your model expects a list
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # Return the result as JSON
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
