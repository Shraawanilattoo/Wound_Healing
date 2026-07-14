import os
from PIL import Image
import numpy as np

# Define the input folder and output folder
input_folder = '/Users/shraawanilattoo/Library/CloudStorage/OneDrive-NorthCarolinaStateUniversity/NCSU/FDS 510/Project Files/Image_data_working/Mouse wound photos-updated/Cropped images'
output_folder = '/Users/shraawanilattoo/Library/CloudStorage/OneDrive-NorthCarolinaStateUniversity/NCSU/FDS 510/Project Files/Image_data_working/Mouse wound photos-updated/green_channel_images'

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Iterate over all the files in the input folder
for filename in os.listdir(input_folder):
    file_path = os.path.join(input_folder, filename)

    # Check if the file is an image (optional: you can add more formats like .png)
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        # Open the image
        image = Image.open(file_path)
        
        # Convert to RGB
        rgb_image = image.convert('RGB')
        
        # Convert the image to a numpy array
        img_array = np.array(rgb_image)
        
        # Set the green and blue channels to 0
        img_array[:, :, 0] = 0  # Red channel
        img_array[:, :, 2] = 0  # Green channel
        
        # Convert the array back to an image
        red_image = Image.fromarray(img_array)
        
        # Define the output path
        output_path = os.path.join(output_folder, filename)
        
        # Save the red channel image
        red_image.save(output_path)
        print(f"Extracted red channel and saved {filename}.")
