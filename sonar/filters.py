import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.restoration import denoise_tv_chambolle, denoise_bilateral, denoise_wavelet, denoise_nl_means, denoise_tv_bregman
from skimage.filters import unsharp_mask
from scipy.stats import wasserstein_distance, ks_2samp
import time

# Function to load and sort images from a directory
def load_images_from_directory(directory, batch_size):
    image_files = sorted([f for f in os.listdir(directory) if f.endswith('.png') or f.endswith('.jpg')])  # Ensure files are sorted
    for i in range(0, len(image_files), batch_size):
        batch_files = image_files[i:i+batch_size]
        batch_images = []
        for image_file in batch_files:
            image_pil = Image.open(os.path.join(directory, image_file))
            batch_images.append(np.array(image_pil.convert('RGB')).astype(np.float32) / 255.0)  # Normalize to [0, 1]
        yield batch_images

# Customizable parameters
batch_size = 1  # Change this to customize batch size
directory = '../datasets/normal2noisy_forest/trainB/'  # Directory containing noisy images
org_directory = '../datasets/gray_forest/trainB/'  # Directory containing corresponding original images

# Initialize lists to accumulate values across the whole dataset
all_emp = []
all_unsharp = []
all_noises_tv_cham = []
all_noises_bi = []
all_noises_wave = []
all_noises_nl = []
all_noises_tv_breg = []
i = 0
# Process the images in batches
for noisy_batch, original_batch in zip(load_images_from_directory(directory, batch_size),
                                       load_images_from_directory(org_directory, batch_size)):
    for noisy, original in zip(noisy_batch, original_batch):
        start = time.time()
        # Calculate emp (noisy / original), handling zeros in original
        emp = np.divide(noisy, original, out=np.zeros_like(original), where=original != 0)
        all_emp.append(emp.flatten())

        epsilon = 1e-3
        noisy = noisy + epsilon
        noisy_log = np.log(noisy)

        # Denoising using Unsharp Masking
        unsharp = unsharp_mask(noisy_log, radius=1, amount=1)
        unsharp = np.exp(unsharp)  # Convert back from log domain
        noises_unsharp = np.divide(noisy, unsharp, out=np.zeros_like(unsharp), where=noisy != 0)
        all_unsharp.append(noises_unsharp.flatten())

        # Denoising using Total Variation (TV) Chambolle
        img_tv = denoise_tv_chambolle(noisy_log, weight=0.1, channel_axis=-1)
        img_tv = np.exp(img_tv)  # Convert back from log domain
        noises_tv = np.divide(noisy, img_tv, out=np.zeros_like(img_tv), where=noisy != 0)
        all_noises_tv_cham.append(noises_tv.flatten())

        # Denoising using Bilateral filter
        img_bi = denoise_bilateral(noisy_log, sigma_color=0.1, sigma_spatial=15, channel_axis=-1)
        img_bi = np.exp(img_bi)  # Convert back from log domain
        noises_bi = np.divide(noisy, img_bi, out=np.zeros_like(img_bi), where=noisy != 0)
        all_noises_bi.append(noises_bi.flatten())

        # Denoising using Wavelet transform
        img_wave = denoise_wavelet(noisy_log, channel_axis=-1, convert2ycbcr=True, rescale_sigma=True)
        img_wave = np.exp(img_wave)
        noises_wave = np.divide(noisy, img_wave, out=np.zeros_like(img_wave), where=noisy != 0)
        all_noises_wave.append(noises_wave.flatten())

        # Denoising using NL Means transform
        img_nl = denoise_nl_means(noisy_log)
        img_nl = np.exp(img_nl)
        noises_nl = np.divide(noisy, img_nl, out=np.zeros_like(img_nl), where=noisy != 0)
        all_noises_nl.append(noises_nl.flatten())

        # Denoising using TV Bregman transform
        img_breg = denoise_tv_bregman(noisy_log)
        img_breg = np.exp(img_breg)
        noises_breg = np.divide(noisy, img_breg, out=np.zeros_like(img_breg), where=noisy != 0)
        all_noises_tv_breg.append(noises_breg.flatten())
        end = time.time()
        print(end - start)
    i = i + 1
    print(i)
    if i == 30:
        break

# Flatten all arrays into one large array for each distribution
all_emp = np.concatenate(all_emp)
all_noises_tv_cham = np.concatenate(all_noises_tv_cham)
all_noises_bi = np.concatenate(all_noises_bi)
all_noises_wave = np.concatenate(all_noises_wave)
all_unsharp = np.concatenate(all_unsharp)
all_noises_nl = np.concatenate(all_noises_nl)
all_noises_tv_breg = np.concatenate(all_noises_tv_breg)

# Plot and save histogram for TV Chambolle denoising
plt.figure(figsize=(8, 6))
plt.hist(all_emp, bins=200, alpha=0.5, label='emp (noisy/original)', color='gray', range=(0, 2.5))
plt.hist(all_noises_tv_cham, bins=200, alpha=0.5, label='tv_chambolle (noisy/denoised)', color='blue', range=(0, 2.5))
plt.title('Histogram of emp vs Total Variation Chambolle (whole dataset)')
plt.xlabel('Ratio')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
# Save the histogram for TV Chambolle
plt.savefig('histogram_tv_chambolle.png')
plt.close()

# Plot and save histogram for Bilateral denoising
plt.figure(figsize=(8, 6))
plt.hist(all_emp, bins=200, alpha=0.5, label='emp (noisy/original)', color='gray', range=(0, 2.5))
plt.hist(all_noises_bi, bins=200, alpha=0.5, label='bilateral (noisy/denoised)', color='blue', range=(0, 2.5))
plt.title('Histogram of emp vs Bilateral (whole dataset)')
plt.xlabel('Ratio')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
# Save the histogram for Bilateral filter
plt.savefig('histogram_bilateral.png')
plt.close()

# Plot and save histogram for Wavelet denoising
plt.figure(figsize=(8, 6))
plt.hist(all_emp, bins=200, alpha=0.5, label='emp (noisy/original)', color='gray', range=(0, 2.5))
plt.hist(all_noises_wave, bins=200, alpha=0.5, label='wavelet (noisy/denoised)', color='blue', range=(0, 2.5))
plt.title('Histogram of emp vs Wavelet (whole dataset)')
plt.xlabel('Ratio')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
# Save the histogram for Wavelet transform
plt.savefig('histogram_wavelet.png')
plt.close()

plt.figure(figsize=(8, 6))
plt.hist(all_emp, bins=200, alpha=0.5, label='emp (noisy/original)', color='gray', range=(0, 2.5))
plt.hist(all_unsharp, bins=200, alpha=0.5, label='unsharp (noisy/denoised)', color='blue', range=(0, 2.5))
plt.title('Histogram of emp vs Unsharp Mask (whole dataset)')
plt.xlabel('Ratio')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
plt.savefig('histogram_unsharp.png')
plt.close()

plt.figure(figsize=(8, 6))
plt.hist(all_emp, bins=200, alpha=0.5, label='emp (noisy/original)', color='gray', range=(0, 2.5))
plt.hist(all_noises_nl, bins=200, alpha=0.5, label='NL Means (noisy/denoised)', color='blue', range=(0, 2.5))
plt.title('Histogram of emp vs NL Means (whole dataset)')
plt.xlabel('Ratio')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
plt.savefig('histogram_nl.png')
plt.close()

plt.figure(figsize=(8, 6))
plt.hist(all_emp, bins=200, alpha=0.5, label='emp (noisy/original)', color='gray', range=(0, 2.5))
plt.hist(all_noises_tv_breg, bins=200, alpha=0.5, label='Bregman (noisy/denoised)', color='blue', range=(0, 2.5))
plt.title('Histogram of emp vs Total Variation Bregman (whole dataset)')
plt.xlabel('Ratio')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
plt.savefig('histogram_breg.png')
plt.close()

# Print Wasserstein distances for the whole dataset
print(f'Total TV Cham EMD: {wasserstein_distance(all_emp, all_noises_tv_cham)}')
print(f'Total Bilateral EMD: {wasserstein_distance(all_emp, all_noises_bi)}')
print(f'Total Wavelet EMD: {wasserstein_distance(all_emp, all_noises_wave)}')
print(f'Total Unsharp EMD: {wasserstein_distance(all_emp, all_unsharp)}')
print(f'Total NL Means EMD: {wasserstein_distance(all_emp, all_noises_nl)}')
print(f'Total TV Bregman EMD: {wasserstein_distance(all_emp, all_noises_tv_breg)}')

# Print KS-test for the whole dataset
print(f'Total TV Cham KS: {ks_2samp(all_emp, all_noises_tv_cham)}')
print(f'Total Bilateral KS: {ks_2samp(all_emp, all_noises_bi)}')
print(f'Total Wavelet KS: {ks_2samp(all_emp, all_noises_wave)}')
print(f'Total Unsharp KS: {ks_2samp(all_emp, all_unsharp)}')
print(f'Total NL Means KS: {ks_2samp(all_emp, all_noises_nl)}')
print(f'Total TV Bregman KS: {ks_2samp(all_emp, all_noises_tv_breg)}')
