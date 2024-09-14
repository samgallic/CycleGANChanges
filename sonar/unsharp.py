import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.restoration import denoise_tv_chambolle, denoise_bilateral, denoise_wavelet
from scipy.stats import wasserstein_distance

# Function to load images from a directory
def load_images_from_directory(directory, batch_size):
    image_files = [f for f in os.listdir(directory) if f.endswith('.png') or f.endswith('.jpg')]
    for i in range(0, len(image_files), batch_size):
        if i > 90:
            break
        batch_files = image_files[i:i+batch_size]
        batch_images = []
        for image_file in batch_files:
            image_pil = Image.open(os.path.join(directory, image_file))
            batch_images.append(np.array(image_pil.convert('RGB')).astype(np.float32) / 255.0)  # Normalize to [0, 1]
        yield batch_images

# Customizable parameters
batch_size = 5  # Change this to customize batch size
directory = '../datasets/normal2noisy_forest/trainB/'  # Directory containing noisy images
org_directory = '../datasets/gray_forest/trainB/'  # Directory containing corresponding original images

# Initialize lists to accumulate values across the whole dataset
all_emp = []
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

        # Denoising using Total Variation (TV) Chambolle
        img_tv = denoise_tv_chambolle(noisy_log, weight=0.1, channel_axis=-1)
        img_tv = np.expm1(img_tv)  # Convert back from log domain
        # img_tv = np.clip(img_tv, 0, 1)
        noises_tv = np.divide(img_tv, original, out=np.zeros_like(img_tv), where=original != 0)
        all_noises_tv.append(noises_tv.flatten())

        # Denoising using Bilateral filter
        img_bi = denoise_bilateral(noisy_log, sigma_color=0.1, sigma_spatial=15, channel_axis=-1)
        img_bi = np.expm1(img_bi)  # Convert back from log domain
        # img_bi = np.clip(img_bi, 0, 1)
        noises_bi = np.divide(img_bi, original, out=np.zeros_like(img_bi), where=original != 0)
        all_noises_bi.append(noises_bi.flatten())

        # Denoising using Wavelet transform
        img_wave = denoise_wavelet(noisy, channel_axis=-1, convert2ycbcr=True, rescale_sigma=True)
        # img_wave = np.clip(img_wave, 0, 1)
        noises_wave = np.divide(img_wave, original, out=np.zeros_like(img_wave), where=original != 0)
        all_noises_wave.append(noises_wave.flatten())

# Flatten all arrays into one large array for each distribution
all_emp = np.concatenate(all_emp)
all_noises_tv = np.concatenate(all_noises_tv)
all_noises_bi = np.concatenate(all_noises_bi)
all_noises_wave = np.concatenate(all_noises_wave)

# Plot and save histogram for the whole dataset
plt.figure(figsize=(8, 6))
plt.hist(all_emp, bins=200, alpha=0.5, label='emp (noisy/original)', color='gray')
plt.hist(all_noises_tv, bins=200, alpha=0.5, label='tv_chambolle (denoised/original)', color='green')
plt.hist(all_noises_bi, bins=200, alpha=0.5, label='bilateral (denoised/original)', color='purple')
plt.hist(all_noises_wave, bins=200, alpha=0.5, label='wavelet (denoised/original)', color='blue')
# plt.ylim(0, 5000)
plt.title('Histogram of emp vs noises (whole dataset)')
plt.xlabel('Ratio')
plt.ylabel('Frequency')
plt.legend(loc='upper right')

# Save the histogram
plt.savefig('histogram_whole_dataset.png')
plt.close()

# Print Wasserstein distances for the whole dataset
print(f'Total TV: {wasserstein_distance(all_emp, all_noises_tv)}')
print(f'Total Bilateral: {wasserstein_distance(all_emp, all_noises_bi)}')
print(f'Total Wavelet: {wasserstein_distance(all_emp, all_noises_wave)}')
