import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from src.data_loader import load_data

def upscale_image():
    model = load_model('custom_model.h5', compile=False)
    
    images, file_names = load_data()
    
    predicted_images = model.predict(images)
    
    for i, predicted_image in enumerate(predicted_images):
        predicted_image = (predicted_image * 255).astype(np.uint8)
        cv2.imwrite(f'data/processed_images/{file_names[i]}', cv2.cvtColor(predicted_image, cv2.COLOR_RGB2BGR))

if __name__ == '__main__':
    upscale_image()
