import pandas as pd
import numpy as np
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os
from sklearn.cluster import MiniBatchKMeans

# Input and output folder paths
input_folder = "/Users/shraawanilattoo/Library/CloudStorage/OneDrive-NorthCarolinaStateUniversity/NCSU/FDS 510/Project Files/Image_data_working/Mouse wound photos-updated/Cropped images normalised"
output_folder = "/Users/shraawanilattoo/Library/CloudStorage/OneDrive-NorthCarolinaStateUniversity/NCSU/FDS 510/Project Files/Image_data_working/Mouse wound photos-updated/Kmeans_output"
os.makedirs(output_folder, exist_ok=True)  # Create output folder if not exists

# Parameters
k = 4  # Number of clusters (adjust as needed)
image_size = (300, 300)  # Resize for faster processing

# Process each image in the folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        image_path = os.path.join(input_folder, filename)
        img = cv2.imread(image_path)
        img_resized = cv2.resize(img, image_size)

        # Reshape image to (num_pixels, 3) for clustering
        pixels = img_resized.reshape(-1, 3)

        # Apply Mini-Batch K-Means
        kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(pixels)

        # Reconstruct segmented image
        segmented_img = kmeans.cluster_centers_[clusters].reshape(img_resized.shape)
        segmented_img = segmented_img.astype(np.uint8)

        # Save segmented image
        output_path = os.path.join(output_folder, f"segmented_{filename}")
        cv2.imwrite(output_path, segmented_img)

        print(f"Processed: {filename}")

print(f"Segmentation completed! Check output folder: {output_folder}")

# -------------------- K-Means Clustering on Histogram Features --------------------
'''
# Load histogram feature data from CSV
csv_hist_path = "/Users/shraawanilattoo/Library/CloudStorage/OneDrive-NorthCarolinaStateUniversity/NCSU/FDS 510/Project Files/Image_data_working/Mouse wound photos-updated/Cropped images normalised/image_hist_features.csv"
df = pd.read_csv(csv_hist_path)

# Extract feature vectors (skip filename column)
hist_features = df.iloc[:, 1:].values  

# Apply K-Means clustering
k = 3  # Number of clusters (adjust as needed)
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(hist_features)

# Display cluster results
print(df[['filename', 'cluster']])

# Save clustered results
df.to_csv("/Users/shraawanilattoo/Library/CloudStorage/OneDrive-NorthCarolinaStateUniversity/NCSU/FDS 510/Project Files/Image_data_working/Mouse wound photos-updated/Cropped images normalised/image_clusters.csv", index=False)

# Plot histogram clusters
plt.scatter(hist_features[:, 0], hist_features[:, 1], c=df['cluster'], cmap='viridis')
plt.title("Image Clustering using K-Means")
plt.show()
'''