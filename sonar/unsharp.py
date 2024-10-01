import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import unsharp_mask
import time

# Function to load and sort images from a directory, yielding images and their filenames
def load_images_from_directory(directory, batch_size):
    image_files = sorted([f for f in os.listdir(directory) if f.endswith('.png') or f.endswith('.jpg')])  # Ensure files are sorted
    for i in range(0, len(image_files), batch_size):
        batch_files = image_files[i:i+batch_size]
        batch_images = []
        for image_file in batch_files:
            image_pil = Image.open(os.path.join(directory, image_file))
            batch_images.append(np.array(image_pil.convert('RGB')).astype(np.float32) / 255.0)  # Normalize to [0, 1]
        yield batch_images, batch_files  # Yield both images and their filenames

# Customizable parameters
batch_size = 16  # Change this to customize batch size
directory = '../datasets/gamma2rayleigh/trainB/'  # Directory containing noisy images
org_directory = '../datasets/gray_forest/trainB/'  # Directory containing corresponding original images
save_directory = '../datasets/unsharp/trainB/'  # Directory to save processed images

# Create the save directory if it doesn't exist
os.makedirs(save_directory, exist_ok=True)

# Initialize lists to accumulate values across the whole dataset
all_emp = []
all_unsharp = []

# Process the images in batches
for (noisy_batch, noisy_filenames), (original_batch, original_filenames) in zip(
    load_images_from_directory(directory, batch_size),
    load_images_from_directory(org_directory, batch_size)
):
    for noisy, original, filename in zip(noisy_batch, original_batch, noisy_filenames):
        start = time.time()
        # Calculate emp (noisy / original), handling zeros in original
        emp = np.divide(noisy, original, out=np.zeros_like(original), where=original != 0)
        all_emp.append(emp.flatten())

        epsilon = 1e-3
        noisy = noisy + epsilon
        noisy_log = np.log(noisy)

        # Denoising using Unsharp Masking
        unsharp = unsharp_mask(noisy_log, radius=1.0, amount=1.0)
        unsharp = np.exp(unsharp)  # Convert back from log domain

        # Convert the processed image to uint8 format
        unsharp_uint8 = (unsharp * 255).astype(np.uint8)
        image = Image.fromarray(unsharp_uint8)

        # Define the path to save the processed image
        save_path = os.path.join(save_directory, filename)
        image.save(save_path)  # Save with the original filename

        # Calculate noises_unsharp
        noises_unsharp = np.divide(noisy, unsharp, out=np.zeros_like(unsharp), where=noisy != 0)
        all_unsharp.append(noises_unsharp.flatten())
        end = time.time()
        print(f"Processed {filename} in {end - start:.2f} seconds")

# Flatten all arrays into one large array for each distribution
all_emp = np.concatenate(all_emp)
all_unsharp = np.concatenate(all_unsharp)

# Plot and save the histogram
plt.figure(figsize=(8, 6))
plt.hist(all_emp, bins=200, alpha=0.5, label='emp (noisy/original)', color='gray', range=(0, 2.5))
plt.hist(all_unsharp, bins=200, alpha=0.5, label='unsharp (noisy/denoised)', color='blue', range=(0, 2.5))
plt.title('Histogram of emp vs Unsharp Mask (whole dataset)')
plt.xlabel('Ratio')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
plt.savefig('rayleigh.png')
plt.close()
