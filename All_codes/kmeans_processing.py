import cv2
import numpy as np
import os
import csv

# Define input and output folders
input_folder = "/Users/shraawanilattoo/Library/CloudStorage/OneDrive-NorthCarolinaStateUniversity/NCSU/FDS 510/Project Files/Image_data_working/Mouse wound photos-updated/Cropped images"
output_folder = "/Users/shraawanilattoo/Library/CloudStorage/OneDrive-NorthCarolinaStateUniversity/NCSU/FDS 510/Project Files/Image_data_working/Mouse wound photos-updated/Cropped images normalised"
os.makedirs(output_folder, exist_ok=True)
'''
# CSV file paths
csv_hist_path = os.path.join(output_folder, "image_hist_features.csv")
csv_pixel_path = os.path.join(output_folder, "image_pixel_features.csv")

# Open CSV file for histogram features
with open(csv_hist_path, "w", newline="") as csvfile_hist:
    csv_writer_hist = csv.writer(csvfile_hist)

    # Open CSV file for pixel data
    with open(csv_pixel_path, "w", newline="") as csvfile_pixel:
        csv_writer_pixel = csv.writer(csvfile_pixel)

        # Histogram bins
        num_bins = 256  
        header_hist = ["filename"] + [f"bin_{i}" for i in range(num_bins * 3)]
        csv_writer_hist.writerow(header_hist)

        header_pixel = ["filename", "row", "col", "R", "G", "B"]
        csv_writer_pixel.writerow(header_pixel)'''

# Process each image in the folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        image_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        # Load image in color
        image = cv2.imread(image_path)

        # Apply Gaussian blur for noise reduction
        denoised_image = cv2.GaussianBlur(image, (9,9), 0)

        # Normalize pixel values to [0,1]
        # normalized_image = denoised_image.astype(np.float32) / 255.0

        # Save processed image
        cv2.imwrite(output_path,denoised_image) #(normalized_image * 255).astype(np.uint8))
'''
                # Compute color histograms
                hist_features = []
                for i in range(3):  # Loop over B, G, R channels
                    hist = cv2.calcHist([image], [i], None, [num_bins], [0, 256])
                    hist = hist.flatten()  # Flatten histogram to 1D
                    hist_features.extend(hist)

                # Normalize histogram features
                hist_features = np.array(hist_features) / sum(hist_features)

                # Write histogram feature vector
                csv_writer_hist.writerow([filename] + hist_features.tolist())

                # Reshape image into a 2D matrix where each row is a pixel with (R, G, B)
                rows, cols, _ = image.shape
                for r in range(rows):
                    for c in range(cols):
                        pixel = image[r, c]  # Get RGB values
                        csv_writer_pixel.writerow([filename, r, c, pixel[2], pixel[1], pixel[0]])  # OpenCV uses BGR, so swap to RGB

print("Processing completed! Check the output folder and CSV files.")'''
