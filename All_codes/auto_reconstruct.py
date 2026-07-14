import matplotlib.pyplot as plt
import torch
from autoencoder1 import model, imgs



model.eval()
with torch.no_grad():
    reconstructed = model(imgs)  # imgs: batch of input images


n = 10  # number of images to display
for i in range(n):
    plt.subplot(2, n, i + 1)
    plt.imshow(imgs[i].permute(1, 2, 0).cpu().numpy())
    plt.title("Original")
    plt.axis("off")

    plt.subplot(2, n, i + 1 + n)
    plt.imshow(reconstructed[i].permute(1, 2, 0).cpu().numpy())
    plt.title("Reconstructed")
    plt.axis("off")

#plt.tight_layout()
plt.show()

