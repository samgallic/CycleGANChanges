import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io

def plot_pixel_histograms(input_dir):
    all_pixels = []

    # Read each image in the directory
    for img_file in os.listdir(input_dir):
        if img_file.endswith(('png', 'jpg', 'jpeg')):
            img_path = os.path.join(input_dir, img_file)
            image = io.imread(img_path)

            # If the image is colored (RGB), flatten it to grayscale
            if len(image.shape) == 3:
                image = image.mean(axis=2)  # Convert to grayscale

            all_pixels.extend(image.flatten())

    # Plot histogram of pixel values
    plt.hist(all_pixels, bins=256, range=(0, 255), density=True, color='gray', alpha=0.75)
    plt.title('Histogram of Pixel Values')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.show()

# Usage
input_directory = '../datasets/unsharp/trainA'
plot_pixel_histograms(input_directory)
