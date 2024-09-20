import numpy as np
import cv2
from scipy.fftpack import fft2, ifft2
from skimage.restoration import denoise_tv_chambolle
import matplotlib.pyplot as plt

# Load image
image = cv2.imread('../datasets/normal2noisy_forest/trainB/Forest-Train (2).png', 0)
original = cv2.imread('../datasets/gray_forest/trainB/Forest-Train (2).png', 0)

# Normalize the image (if needed)
image = image / 255.0

# Step 1: Fourier Transform
image_fft = fft2(image)

# Step 2: Logarithmic transformation
log_image_fft = np.log(np.abs(image_fft) + 1e-8)

# Step 3: Apply Denoising (e.g., Total Variation denoising)
denoised_log_image = denoise_tv_chambolle(log_image_fft, weight=0.1)

# Step 4: Exponential Transform
exp_denoised_image = np.exp(denoised_log_image)

# Step 5: Inverse Fourier Transform
# Since fft2 gives complex output, we reconstruct using real part
reconstructed_image = np.abs(ifft2(exp_denoised_image))

# Step 6: Normalize the reconstructed image for display
reconstructed_image = (reconstructed_image - np.min(reconstructed_image)) / \
                      (np.max(reconstructed_image) - np.min(reconstructed_image))

# Display the original and denoised images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Noisy Image')

plt.subplot(1, 2, 2)
plt.imshow(reconstructed_image, cmap='gray')
plt.title('Reconstructed Denoised Image')

plt.show()