import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance

def convert_images_to_greyscale(folder_path):
    # Loop through each file in the directory
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # Check if it's an image (you can add more extensions if needed)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Read the image
            img = cv2.imread(file_path)
            
            # Convert to grayscale
           # grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Save the grayscale image with the same name
            cv2.imwrite(file_path, img)

# Example usage
folder_path = '/Users/shraawanilattoo/Library/CloudStorage/OneDrive-NorthCarolinaStateUniversity/NCSU/FDS 510/Project Files/Image_data_working/Mouse wound photos-updated/Cropped images adjusted brightness'  # Replace with your folder path
convert_images_to_greyscale(folder_path)


def adjust_brightness(image, target_brightness=128):
    current_brightness = np.mean(image)  # Compute mean brightness
    alpha = target_brightness / current_brightness  # Compute scaling factor
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
    return adjusted

def process_images(folder_path, target_brightness=128):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        img = cv2.imread(file_path) #cv2.IMREAD_GRAYSCALE)
        adjusted_img = adjust_brightness(img, target_brightness)
        cv2.imwrite(file_path, adjusted_img)

process_images(folder_path)


def image_to_vector(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        img = cv2.imread(file_path) #cv2.IMREAD_GRAYSCALE)
        
        # Flatten the image into a 1D vector
        img_vector = img.flatten()

    return img_vector

vector = image_to_vector(folder_path)
print(vector)

saturation_factor = 0.50
for filename in os.listdir(folder_path):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(folder_path, filename)
        img = Image.open(img_path).convert("RGB")

        enhancer = ImageEnhance.Color(img)
        saturated_img = enhancer.enhance(saturation_factor)

        # Save adjusted image
        output_path = os.path.join(folder_path, filename)
        saturated_img.save(output_path)