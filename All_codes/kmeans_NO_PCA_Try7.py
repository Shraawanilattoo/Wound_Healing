import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
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
print(image_vectors.shape)

# Normalize image vectors to [0, 1]
image_vectors = image_vectors / 255.0

# Apply K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)  # Change '4' based on your dataset
labels = kmeans.fit_predict(image_vectors)  # Clustering

# Scatter plot with K-means clustering result (if it's a 2D feature space, you can plot)
plt.figure(figsize=(10, 8))
scatter = plt.scatter(image_vectors[:, 0], image_vectors[:, 1], c=labels, cmap='viridis', s=50, alpha=0.7)

# Add labels to each point (image filenames)
for i, filename in enumerate(os.listdir(folder_path)):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        # Only annotate the points that correspond to images
        plt.annotate(filename, (image_vectors[i, 0], image_vectors[i, 1]), 
                     textcoords="offset points", xytext=(0, 5), ha='center', fontsize=8)

# Customize plot
plt.colorbar(scatter, label='Cluster Label')
plt.xlabel("Pixel 1")
plt.ylabel("Pixel 2")
plt.title("K-Means Clustering of Images")
plt.grid(True)

# Show the plot
plt.show()

# Optionally, you can display clustered images
for cluster_id in range(kmeans.n_clusters):
    cluster_images = [images[i] for i in range(len(labels)) if labels[i] == cluster_id]
    
    plt.figure(figsize=(10, 8))
    for i, img in enumerate(cluster_images[:9]):  # Display the first 5 images from each cluster
        plt.subplot(1,9,i+1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Cluster {cluster_id}")
    plt.show()
