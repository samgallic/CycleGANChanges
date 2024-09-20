from scipy.ndimage import uniform_filter
from scipy.ndimage import variance
from skimage.filters import unsharp_mask
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from scipy.stats import wasserstein_distance, ks_2samp

def load_images_from_directory(directory, batch_size):
    image_files = sorted([f for f in os.listdir(directory) if f.endswith('.png') or f.endswith('.jpg')])  # Ensure files are sorted
    for i in range(0, len(image_files), batch_size):
        batch_files = image_files[i:i+batch_size]
        batch_images = []
        for image_file in batch_files:
            image_pil = Image.open(os.path.join(directory, image_file)).convert('L')  # Convert to grayscale
            batch_images.append(np.array(image_pil).astype(np.float32) / 255.0)  # Normalize to [0, 1]
        yield batch_images

def lee_filter(img, size):
    img_mean = uniform_filter(img, size)
    img_sqr_mean = uniform_filter(img**2, size)
    img_variance = img_sqr_mean - img_mean**2

    overall_variance = variance(img)

    img_weights = img_variance / (img_variance + overall_variance)
    img_output = img_mean + img_weights * (img - img_mean)
    return img_output

def main():
    directory = '../datasets/normal2noisy_forest/trainB/'
    org_directory = '../datasets/gray_forest/trainB/'
    batch_size = 64
    all_emp = []
    all_lee = []
    all_unsharp = []
    i = 0
    for noisy_batch, original_batch in zip(load_images_from_directory(directory, batch_size),
                                          load_images_from_directory(org_directory, batch_size)):
        for noisy, original in zip(noisy_batch, original_batch):
            epsilon = 1e-3
            noisy = noisy + epsilon
            noisy_log = np.log(noisy)

            unsharp = unsharp_mask(noisy_log)
            # unsharp = np.exp(unsharp)

            unnoised = lee_filter(noisy_log, 100)
            # unnoised = np.log(unnoised + epsilon)

            emp = noisy_log - np.log(original + epsilon)
            lee = noisy_log - unnoised
            uns = noisy_log - unsharp

            all_emp.append(emp)
            all_lee.append(lee)
            all_unsharp.append(uns)

    # Flatten the arrays
    all_emp = np.concatenate(all_emp).ravel()
    all_lee = np.concatenate(all_lee).ravel()
    all_unsharp = np.concatenate(all_unsharp).ravel()
    print(len(all_lee))
    print(len(all_unsharp))
    all_emp = all_emp[~np.isnan(all_emp) & ~np.isinf(all_emp)]
    all_lee = all_lee[~np.isnan(all_lee) & ~np.isinf(all_lee)]
    all_unsharp = all_unsharp[~np.isnan(all_unsharp) & ~np.isinf(all_unsharp)]
    print(len(all_lee))
    print(len(all_unsharp))

    print('Lee EMD ', wasserstein_distance(all_emp, all_lee))
    print('Lee KS ', ks_2samp(all_emp, all_lee))
    print('Unsharp EMD ', wasserstein_distance(all_emp, all_unsharp))
    print('Unsharp KS ', ks_2samp(all_emp, all_unsharp))

    # Plot histograms
    plt.figure(figsize=(8, 6))
    plt.hist(all_emp, bins=200, alpha=0.5, label='emp (noisy/original)', color='gray')
    plt.hist(all_lee, bins=200, alpha=0.5, label='lee (noisy/denoised)', color='blue')
    plt.title('Histogram of emp vs Lee Filter (whole dataset)')
    plt.xlabel('Ratio')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    plt.savefig('histogram_lee.png')
    plt.close()

    # Plot histograms
    plt.figure(figsize=(8, 6))
    plt.hist(all_emp, bins=200, alpha=0.5, label='emp (noisy/original)', color='gray')
    plt.hist(all_unsharp, bins=200, alpha=0.5, label='unsharp (noisy/denoised)', color='blue')
    plt.title('Histogram of emp vs Unsharp Masking (whole dataset)')
    plt.xlabel('Ratio')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    plt.savefig('histogram_unsharp.png')
    plt.close()

if __name__ == '__main__':
    main()