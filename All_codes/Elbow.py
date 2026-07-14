import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from kmeans_PCA_Try6 import *

# Elbow Method for finding optimal number of clusters
def elbow_method(image_vectors_pca, max_k=10):
    distortions = []
    
    # Loop over a range of K values to calculate inertia (distortion)
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(image_vectors_pca)
        distortions.append(kmeans.inertia_)
    
    # Plot the distortion for each value of K
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, max_k + 1), distortions, marker='o', linestyle='-', color='b')
    plt.title('Elbow Method For Optimal K')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Distortion (Inertia)')
    plt.xticks(range(1, max_k + 1))
    plt.grid(True)
    plt.show()

# Call the elbow method with the PCA-transformed image vectors and max K value (e.g., 10)
elbow_method(image_vectors_pca, max_k=10)
