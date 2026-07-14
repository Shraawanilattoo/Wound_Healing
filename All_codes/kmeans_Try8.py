import os
import numpy as np
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Path to the folder containing images
image_folder = '/Users/shraawanilattoo/Library/CloudStorage/OneDrive-NorthCarolinaStateUniversity/NCSU/FDS 510/Project Files/Short Data'

# Function to read images and convert to a list of feature vectors
def load_images(image_folder):
    image_data = []
    for filename in os.listdir(image_folder):
        img_path = os.path.join(image_folder, filename)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (64, 64))  # Resize to a fixed size
        img = img.flatten()  # Flatten the image to a 1D array
        image_data.append(img)
    return np.array(image_data)

# Load images
X = load_images(image_folder)
print(X)
# Perform K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)  # Change n_clusters as needed
kmeans.fit(X)

# Plot results (example)
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_)
plt.show()
