import cv2
import numpy as np
import os

def preprocess_image(img):
    # Apply Gaussian blur to deblur the image
    blurred_img = cv2.GaussianBlur(img, (5, 5), 0)
    
    # Convert RGB image to LAB color space
    lab_img = cv2.cvtColor(blurred_img, cv2.COLOR_RGB2LAB)
    
    # Split the LAB image into L, A, and B channels
    l, a, b = cv2.split(lab_img)
    
    # Apply CLAHE to the L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l)
    
    # Merge the enhanced L channel with the original A and B channels
    enhanced_lab_img = cv2.merge((enhanced_l, a, b))
    
    # Convert enhanced LAB image back to RGB
    enhanced_img = cv2.cvtColor(enhanced_lab_img, cv2.COLOR_LAB2RGB)
    
    return enhanced_img

def load_data(input_dir, output_dir, target_size=(256, 256)):
    input_images = []
    output_images = []
    
    files = os.listdir(input_dir)
    
    for file_name in files:
        if file_name.endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(input_dir, file_name)
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, target_size)
                
                high_res_path = os.path.join(output_dir, file_name)
                if os.path.exists(high_res_path):
                    high_res_img = cv2.imread(high_res_path)
                    high_res_img = cv2.cvtColor(high_res_img, cv2.COLOR_BGR2RGB)
                    high_res_img = cv2.resize(high_res_img, target_size)
                    
                    input_images.append(img)
                    output_images.append(high_res_img)
    
    return np.array(input_images), np.array(output_images)

# Define directories
input_dir = 'data/raw_images'
output_dir = 'data/processed_images'

# Load images
X_train, y_train = load_data(input_dir, output_dir)

# Normalize images
X_train = X_train.astype('float32') / 255.0
y_train = y_train.astype('float32') / 255.0
