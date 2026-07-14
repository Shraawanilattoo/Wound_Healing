import cv2
import numpy as np
import os

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

folder_path = '/Users/shraawanilattoo/Library/CloudStorage/OneDrive-NorthCarolinaStateUniversity/NCSU/FDS 510/Project Files/Image_data_working/Mouse wound photos-updated/red_channel_images'  # Replace with actual path
images = load_and_preprocess_images(folder_path)

def images_to_vectors(images):
    num_samples = images.shape[0]
    return images.reshape(num_samples, -1)  # Flatten images

image_vectors = images_to_vectors(images)
print(image_vectors.shape)

image_vectors = image_vectors / 255.0  # Normalize to [0,1] range

from sklearn.decomposition import PCA

pca = PCA(n_components=10)  # Reduce to 50 principal components
image_vectors_pca = pca.fit_transform(image_vectors)


# K-means

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=4, random_state=42)  # Change '5' based on Elbow Method
labels = kmeans.fit_predict(image_vectors_pca)  # Clustering

print(labels)  # Each image gets a cluster label

import matplotlib.pyplot as plt

plt.scatter(image_vectors_pca[:, 0], image_vectors_pca[:, 1], c=labels, cmap='viridis')
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("K-Means Clustering of Images")
plt.show()
