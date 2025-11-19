import os
import logging
import joblib
import json
import numpy as np

#Set up logging
logging.basicConfig(level=logging.INFO)

def init():
    #initialize the model when the container starts

    global model, vectorizer

    try:
        model_dir = os.getenv('AZUREML_MODEL_DIR')
        logging.info(f"Model directory: {model_dir}")

        #list files to see what's available
        if model_dir and os.path.exists(model_dir):
            logging.info(f"Files in model directory: {os.listdir(model_dir)}")

        #load model - adjust path based on whats above
        model_path = os.path.join(model_dir, 'bank_model.pkl')

        if not os.path.exists(model_path):
            #fallback: try current directory
            model_path = 'bank_model.pkl'
            logging.info(f"Trying current directory for model path")
        logging.info(f"Loading model from: {model_path}")

        #load model using joblib (more reliable than pickle for sklearn models)
        with open(model_path, 'rb') as f:
            #the model is stored as a tuple (vectorizer, model)
            vectorizer, model = joblib.load(f)

        logging.info("Model loaded successfully.")
        logging.info(f"Model type: {type(model)}")
        logging.info(f"Vectorizer type: {type(vectorizer)}")

    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        raise

def run(raw_data):
    #handle prediction requests
    try:
        logging.info(f"Received prediction request")

        #parse input data
        data = json.loads(raw_data)
        #get text from request - support different input formats
        text = data.get('text', "")
        if not text:
            #Alternative: check if data is directly the text
            if isinstance(data, str):
                text = data
            else:
                return {"error": "No texts provided in 'text' field."}

        logging.info(f"Processing texts: {text}")

        #transform and predict
        X = vectorizer.transform([text])
        prediction = model.predict(X)[0]

        #get probabilities if available
        probabilities = []
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X)[0].tolist()

        #return result
        result = {
            "prediction": str(prediction),
            "probabilities": probabilities
        }
        
        logging.info(f"Prediction: {result}")
        return result
    
    except Exception as e:
        error_msg = f"Prediction failed: {str(e)}"
        logging.error(error_msg)
        return {"error": error_msg}