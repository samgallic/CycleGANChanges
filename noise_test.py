import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Directories containing the images
noisy_dir = 'datasets/gamma2rayleigh/trainB'
clean_dir = 'datasets/gray_forest/trainB'

# Pixel value ranges
ranges = [
    (0, 50),
    (50, 100),
    (100, 150),
    (150, 200),
    (200, 255)
]

# Initialize lists to store noise values for each range
noise_values = [[] for _ in ranges]

# Get sorted lists of filenames
noisy_filenames = sorted(os.listdir(noisy_dir))
clean_filenames = sorted(os.listdir(clean_dir))

# Ensure that the number of files matches
if len(noisy_filenames) != len(clean_filenames):
    print("Mismatch in the number of files between the noisy and clean datasets.")
    exit()

# Iterate over the filenames
for noisy_filename, clean_filename in zip(noisy_filenames, clean_filenames):
    if noisy_filename != clean_filename:
        print(f"Filename mismatch: {noisy_filename} vs {clean_filename}")
        continue  # Skip mismatched files

    # Load images
    noisy_image_path = os.path.join(noisy_dir, noisy_filename)
    clean_image_path = os.path.join(clean_dir, clean_filename)

    noisy_image = Image.open(noisy_image_path).convert('L')  # Convert to grayscale
    clean_image = Image.open(clean_image_path).convert('L')

    # Convert images to NumPy arrays
    noisy_array = np.array(noisy_image, dtype=np.float32)
    clean_array = np.array(clean_image, dtype=np.float32)

    # Check if image shapes match
    if noisy_array.shape != clean_array.shape:
        print(f"Shape mismatch for file {noisy_filename}. Skipping.")
        continue

    # Calculate the noise (difference)
    noise = noisy_array - clean_array

    # Collect noise values based on the original pixel value ranges
    for idx, (lower, upper) in enumerate(ranges):
        if idx == 0:
            mask = (clean_array >= lower) & (clean_array <= upper)
        else:
            mask = (clean_array > lower) & (clean_array <= upper)
        noise_in_range = noise[mask]
        noise_values[idx].extend(noise_in_range.flatten())

# Create and save histograms
for idx, (lower, upper) in enumerate(ranges):
    plt.figure(figsize=(8, 6))
    plt.hist(noise_values[idx], bins=100, range=(-255, 255), density=True, color='blue', alpha=0.7)
    plt.title(f'Noise Distribution for Original Pixel Values in ({lower}, {upper}]')
    plt.xlabel('Noise Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()
    # Save the histogram
    histogram_filename = f'histogram_{lower}_{upper}.png'
    plt.savefig(histogram_filename)
    plt.close()
    print(f"Saved histogram: {histogram_filename}")
