import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Function to load and preprocess images
def load_and_preprocess_images(folder_path, img_size=(64, 64)):
    image_vectors = []
    
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)  # Load image in color (BGR)

            if img is None:
                continue

            img = cv2.resize(img, img_size)  # Resize to fixed dimensions
            image_vectors.append(img)

    return np.array(image_vectors)

# Path to the folder with images
folder_path = '/Users/shraawanilattoo/Library/CloudStorage/OneDrive-NorthCarolinaStateUniversity/NCSU/FDS 510/Project Files/Short Data'  # Replace with actual path
images = load_and_preprocess_images(folder_path)

# Function to flatten images
def images_to_vectors(images):
    num_samples = images.shape[0]
    return images.reshape(num_samples, -1)  # Flatten images

image_vectors = images_to_vectors(images)
print(image_vectors)

# Normalize image vectors to [0, 1]
image_vectors = image_vectors / 255.0
print(image_vectors)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=20)  # Reduce to 50 principal components
image_vectors_pca = pca.fit_transform(image_vectors)

# Apply K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)  # Change '4' based on the Elbow Method
labels = kmeans.fit_predict(image_vectors_pca)  # Clustering

# Scatter plot with K-means clustering result
plt.figure(figsize=(10, 8))
scatter = plt.scatter(image_vectors_pca[:, 0], image_vectors_pca[:, 1], c=labels, cmap='viridis', s=50, alpha=0.7)

# Add labels to each point (image filenames)
for i, filename in enumerate(os.listdir(folder_path)):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        # Only annotate the points that correspond to images
        plt.annotate(filename, (image_vectors_pca[i, 0], image_vectors_pca[i, 1]), 
                     textcoords="offset points", xytext=(0, 5), ha='center', fontsize=8)

# Customize plot
plt.colorbar(scatter, label='Cluster Label')
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("K-Means Clustering of Images")
plt.grid(True)

# Show the plot
plt.show()
