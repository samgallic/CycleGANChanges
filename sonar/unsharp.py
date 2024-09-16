import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.restoration import denoise_tv_chambolle, denoise_bilateral, denoise_wavelet
from skimage.filters import unsharp_mask
from scipy.stats import wasserstein_distance

# Function to load images from a directory
def load_images_from_directory(directory, batch_size):
    image_files = [f for f in os.listdir(directory) if f.endswith('.png') or f.endswith('.jpg')]
    for i in range(0, len(image_files), batch_size):
        batch_files = image_files[i:i+batch_size]
        batch_images = []
        for image_file in batch_files:
            image_pil = Image.open(os.path.join(directory, image_file))
            batch_images.append(np.array(image_pil.convert('RGB')).astype(np.float32) / 255.0)  # Normalize to [0, 1]
        yield batch_images

# Customizable parameters
batch_size = 16  # Change this to customize batch size
directory = '../datasets/normal2noisy_forest/trainB/'  # Directory containing noisy images
org_directory = '../datasets/gray_forest/trainB/'  # Directory containing corresponding original images

# Initialize lists to accumulate values across the whole dataset
all_emp = []
all_unsharp = []
all_noises_tv = []
all_noises_bi = []
all_noises_wave = []
i = 0
# Process the images in batches
for noisy_batch, original_batch in zip(load_images_from_directory(directory, batch_size),
                                       load_images_from_directory(org_directory, batch_size)):
    for noisy, original in zip(noisy_batch, original_batch):
        # Calculate emp (noisy / original), handling zeros in original
        emp = np.divide(noisy, original, out=np.zeros_like(noisy), where=original != 0)
        all_emp.append(emp.flatten())

        # Logarithmic transformation for TV and Bilateral denoising
        epsilon = 1e-3
        noisy_log = np.log(noisy + epsilon)

        unsharp = unsharp_mask(noisy_log, radius=1, amount=1)
        unsharp = np.exp(unsharp)  # Convert back from log domain
        noises_unsharp = np.divide(unsharp, original, out=np.zeros_like(unsharp), where=original != 0)
        all_unsharp.append(noises_unsharp.flatten())

        # Denoising using Total Variation (TV) Chambolle
        img_tv = denoise_tv_chambolle(noisy_log, weight=0.1, channel_axis=-1)
        img_tv = np.exp(img_tv)  # Convert back from log domain
        # img_tv = np.clip(img_tv, 0, 1)
        noises_tv = np.divide(img_tv, original, out=np.zeros_like(img_tv), where=original != 0)
        all_noises_tv.append(noises_tv.flatten())

        # Denoising using Bilateral filter
        img_bi = denoise_bilateral(noisy_log, sigma_color=0.1, sigma_spatial=15, channel_axis=-1)
        img_bi = np.exp(img_bi)  # Convert back from log domain
        # img_bi = np.clip(img_bi, 0, 1)
        noises_bi = np.divide(img_bi, original, out=np.zeros_like(img_bi), where=original != 0)
        all_noises_bi.append(noises_bi.flatten())

        # Denoising using Wavelet transform
        img_wave = denoise_wavelet(noisy_log, channel_axis=-1, convert2ycbcr=True, rescale_sigma=True)
        img_wave = np.exp(img_wave)
        # img_wave = np.clip(img_wave, 0, 1)
        noises_wave = np.divide(img_wave, original, out=np.zeros_like(img_wave), where=original != 0)
        all_noises_wave.append(noises_wave.flatten())

# Flatten all arrays into one large array for each distribution
all_emp = np.concatenate(all_emp)
all_noises_tv = np.concatenate(all_noises_tv)
all_noises_bi = np.concatenate(all_noises_bi)
all_noises_wave = np.concatenate(all_noises_wave)
all_unsharp = np.concatenate(all_unsharp)

# Plot and save histogram for TV Chambolle denoising
plt.figure(figsize=(8, 6))
plt.hist(all_emp, bins=200, alpha=0.5, label='emp (noisy/original)', color='gray', range=(0, 10))
plt.hist(all_noises_tv, bins=200, alpha=0.5, label='tv_chambolle (denoised/original)', color='green', range=(0, 10))
plt.title('Histogram of emp vs TV Chambolle (whole dataset)')
plt.xlabel('Ratio')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
# Save the histogram for TV Chambolle
plt.savefig('histogram_tv_chambolle.png')
plt.close()

# Plot and save histogram for Bilateral denoising
plt.figure(figsize=(8, 6))
plt.hist(all_emp, bins=200, alpha=0.5, label='emp (noisy/original)', color='gray', range=(0, 10))
plt.hist(all_noises_bi, bins=200, alpha=0.5, label='bilateral (denoised/original)', color='purple', range=(0, 10))
plt.title('Histogram of emp vs Bilateral (whole dataset)')
plt.xlabel('Ratio')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
# Save the histogram for Bilateral filter
plt.savefig('histogram_bilateral.png')
plt.close()

# Plot and save histogram for Wavelet denoising
plt.figure(figsize=(8, 6))
plt.hist(all_emp, bins=200, alpha=0.5, label='emp (noisy/original)', color='gray', range=(0, 10))
plt.hist(all_noises_wave, bins=200, alpha=0.5, label='wavelet (denoised/original)', color='blue', range=(0, 10))
plt.title('Histogram of emp vs Wavelet (whole dataset)')
plt.xlabel('Ratio')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
# Save the histogram for Wavelet transform
plt.savefig('histogram_wavelet.png')
plt.close()

# Plot and save histogram for Unsharp Mask denoising
plt.figure(figsize=(8, 6))
plt.hist(all_emp, bins=200, alpha=0.5, label='emp (noisy/original)', color='gray', range=(0, 10))
plt.hist(all_unsharp, bins=200, alpha=0.5, label='unsharp (denoised/original)', color='orange', range=(0, 10))
plt.title('Histogram of emp vs Unsharp Mask (whole dataset)')
plt.xlabel('Ratio')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
# Save the histogram for Unsharp Mask
plt.savefig('histogram_unsharp.png')
plt.close()

# Print Wasserstein distances for the whole dataset
print(f'Total TV: {wasserstein_distance(all_emp, all_noises_tv)}')
print(f'Total Bilateral: {wasserstein_distance(all_emp, all_noises_bi)}')
print(f'Total Wavelet: {wasserstein_distance(all_emp, all_noises_wave)}')
print(f'Total Unsharp: {wasserstein_distance(all_emp, all_unsharp)}')

