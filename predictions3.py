#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
import numpy as np
from PIL import Image

# Load the pre-trained model
# model_path = 'vgg16_functional_model.h5'  # Update with the correct path if needed
# model = load_model(model_path)

# Function to preprocess the input image
def preprocess_image(image_path, target_size=(180, 180)):
    """
    Preprocess the input image to fit the model's input requirements.
    
    :param image_path: Path to the input image file
    :param target_size: Tuple indicating the target size (default is (180, 180))
    :return: Preprocessed image ready for prediction
    """
    # Load image
    img = Image.open(image_path)
    
    # Convert image to RGB (this handles grayscale images)
    img = img.convert('RGB')
    
    # Resize to target size
    img = img.resize(target_size)
    
    # Convert image to numpy array
    img_array = np.array(img)
    
    # Expand dimensions to match model input shape (1, 180, 180, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Normalize the image to range [0, 1]
    img_array = img_array / 255.0
    
    return img_array

# Function to classify the prediction
def get_binary_result(prediction_value, threshold=0.5):
    """
    Classify the prediction value as 'yes' or 'no' based on the threshold.
    
    :param prediction_value: The prediction value from the model
    :param threshold: The threshold for classification (default is 0.5)
    :return: 'yes' or 'no' based on the prediction value
    """
    if prediction_value > threshold:
        return "yes"
    else:
        return "no"

# Original function to perform inference
def predict_image(model, image_path):
    """
    Perform inference on the input image and return predictions.
    
    :param model: Loaded Keras model
    :param image_path: Path to the input image file
    :return: Predictions (class probabilities or labels)
    """
    # Preprocess the image
    processed_image = preprocess_image(image_path)
    
    # Run the model to get predictions
    predictions = model.predict(processed_image)
    
    return predictions

# Function to get the binary result based on predictions
def get_prediction_result(model, image_path):
    """
    Get the binary classification result based on the predictions.
    
    :param model: Loaded Keras model
    :param image_path: Path to the input image file
    :return: Predictions and the binary result ('yes' or 'no')
    """
    # Get the predictions using the original predict_image function
    predictions = predict_image(model, image_path)
    
    # Extract the prediction value
    prediction_value = predictions[0][0]  # Get the single prediction value

    # Get the binary result using the separate function
    result = get_binary_result(prediction_value)

    return predictions, result  # Return predictions and the classification result
