import cv2
import numpy as np
import os

def load_data(input_dir, output_dir, img_size=(64, 64)):
    input_images = []
    output_images = []
    
    files = os.listdir(input_dir)
    
    for file_name in files:
        if file_name.endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(input_dir, file_name)
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, img_size)
                
                # Assuming the high-resolution images are in the output directory
                high_res_path = os.path.join(output_dir, file_name)
                if os.path.exists(high_res_path):
                    high_res_img = cv2.imread(high_res_path)
                    high_res_img = cv2.cvtColor(high_res_img, cv2.COLOR_BGR2RGB)
                    high_res_img = cv2.resize(high_res_img, img_size)
                    
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
