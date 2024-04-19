import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2

def upscale_image():
    # Load trained model
    model = load_model('custom_model.h5')
    
    # Load and preprocess input image
    input_image = cv2.imread('data/raw_images/0001x2.png')
    input_image = cv2.resize(input_image, (256, 256))  # Resize to match model input shape
    input_image = np.expand_dims(input_image, axis=0)  # Add batch dimension
    input_image = input_image / 255.0  # Normalize
    
    # Make prediction
    predicted_image = model.predict(input_image)
    
    # Post-process predicted image
    predicted_image = (predicted_image * 255).astype(np.uint8)
    predicted_image = predicted_image.reshape((predicted_image.shape[1], predicted_image.shape[2], 3))
    
    # Save predicted image
    cv2.imwrite('data/processed_images/0001x2.png', predicted_image)

if __name__ == '__main__':
    upscale_image()
