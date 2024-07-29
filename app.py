@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the data from the POST request
        data = request.get_json()
        
        # Check if the input key exists in the request data
        if not data or 'input' not in data:
            logging.warning('Input data missing or incorrect format.')
            return jsonify({'error': 'No input data provided'}), 400
        
        # Extract input data
        input_data = data['input']
        
        # Validate input data type
        if not isinstance(input_data, list):
            logging.warning('Input data type is incorrect.')
            return jsonify({'error': 'Input data should be a list'}), 400
        
        # Ensure input data is not empty
        if not input_data:
            logging.warning('Input data list is empty.')
            return jsonify({'error': 'Input data list is empty'}), 400
        
        # Make prediction
        prediction = model.predict(input_data)  # Remove the outer list

        # Return the result as JSON
        return jsonify({'prediction': prediction.tolist()})
    
    except Exception as e:
        logging.error(f"Error during prediction: {e}", exc_info=True)
        return jsonify({'error': 'An error occurred during prediction'}), 500
