import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from Try6 import *


# Reconstruct the images from the PCA components
reconstructed_vectors = pca.inverse_transform(image_vectors_pca)

# Reshape the reconstructed vectors back to the original image shape
reconstructed_images = reconstructed_vectors.reshape(images.shape)

# Display the reconstructed images (for example, the first 5 images)
for i in range(5):
    plt.imshow(reconstructed_images[i])
    plt.title(f'Reconstructed Image {i+1}')
    plt.axis('off')
    plt.show()
