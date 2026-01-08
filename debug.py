#!/usr/bin/env python3
"""Debug script to check PSLG structure."""

import numpy as np
import cv2
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

# Load image
image = cv2.imread("input.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Denoise
denoised = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

# Quantize
h, w, c = denoised.shape
pixels = denoised.reshape(-1, 3).astype(np.float32)
kmeans = MiniBatchKMeans(n_clusters=48, random_state=42, batch_size=1000, n_init=3)
labels = kmeans.fit_predict(pixels)
centers = kmeans.cluster_centers_.astype(np.uint8)
quantized_pixels = centers[labels]
quantized = quantized_pixels.reshape(h, w, c)

# Save quantized image
cv2.imwrite("debug_quantized.png", cv2.cvtColor(quantized, cv2.COLOR_RGB2BGR))

# Extract one color region as a test
test_color = centers[0]
mask = np.all(quantized == test_color, axis=2).astype(np.uint8)

# Find contours
contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

print(f"Number of contours for test color: {len(contours)}")

# Visualize
plt.figure(figsize=(15, 5))

plt.subplot(131)
plt.imshow(image)
plt.title("Original")
plt.axis('off')

plt.subplot(132)
plt.imshow(quantized)
plt.title("Quantized")
plt.axis('off')

plt.subplot(133)
plt.imshow(mask, cmap='gray')
plt.title("Test color mask")
plt.axis('off')

plt.tight_layout()
plt.savefig("debug_viz.png", dpi=150, bbox_inches='tight')
print("Saved debug_viz.png")
