import numpy as np
import cv2
import os

def load_data():
    # Define the path to the raw images directory
    raw_images_dir = 'data/raw_images/'
    
    # Get the list of image filenames in the raw images directory
    image_filenames = [os.path.join(raw_images_dir, filename) for filename in os.listdir(raw_images_dir) if filename.endswith('.png')]
    
    # Initialize lists to store images and labels
    images = []
    labels = []
    
    # Loop through the image filenames
    for filename in image_filenames:
        # Read the image using OpenCV
        image = cv2.imread(filename)
        
        # Resize the image to match the input shape of the model (64x64)
        image = cv2.resize(image, (256, 256))
        
        # Normalize the pixel values to the range [0, 1]
        image = image / 255.0
        
        # Append the image to the images list
        images.append(image)
        
        # Assuming the label is the same as the input image (change as per your dataset)
        labels.append(image)
    
    # Convert the lists to NumPy arrays
    images = np.array(images)
    labels = np.array(labels)
    
    return images, labels
