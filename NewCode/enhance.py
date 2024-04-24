import cv2
import os

# Input and output directories
input_dir = 'NewCode/input/'
output_dir = 'NewCode/output/'

# Ensure output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# List all files in input directory
files = os.listdir(input_dir)

# Loop through each file and process
for file_name in files:
    # Check if file is an image (you can expand this check based on file extensions)
    if file_name.endswith(('.jpg', '.jpeg', '.png', '.gif')):
        # Read the image using OpenCV
        img_path = os.path.join(input_dir, file_name)
        
        # Check if the file exists
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            
            # Check if the image is valid
            if img is not None:
                # Upscale the image to 1080px using bicubic interpolation
                upscaled = cv2.resize(img, (1080, 1080), interpolation=cv2.INTER_CUBIC)
                
                # Save the processed image to output directory
                cv2.imwrite(os.path.join(output_dir, file_name), upscaled)
                
                print(f"Processed {file_name} and saved to {output_dir}")
            else:
                print(f"Error reading {file_name}. Skipping...")
        else:
            print(f"File {file_name} not found. Skipping...")
