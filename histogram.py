import os
import torchvision
from torchvision.utils import save_image
import torch
import matplotlib.pyplot as plt
from scipy.stats import kstest, rayleigh
import numpy as np
from torchvision import io, transforms
from PIL import Image
from torch.nn.functional import normalize
import pickle

def plot_pdf_with_rayleigh(data, title):
    if isinstance(data, np.ndarray):
        flat_data = data.flatten()
        x = np.linspace(0, np.max(flat_data), 1000)
    elif isinstance(data, list) and isinstance(data[0], np.ndarray):
        flat_data = np.concatenate([d.flatten() for d in data])
        x = np.linspace(0, np.max(flat_data), 1000)
    else:
        flat_data = torch.cat([d.flatten() for d in data])
        flat_data = flat_data[torch.isfinite(flat_data)]
        x = torch.linspace(0, torch.max(flat_data).item(), 1000)

    print(f"Noise Values - Min: {flat_data.min()}, Max: {flat_data.max()}, Mean: {flat_data.mean()}")

    plt.figure(figsize=(10, 6))
    plt.hist(flat_data, bins=200, color='blue', alpha=0.7, density=True, label='Noise Data')

    rayleigh_pdf = rayleigh.pdf(x, scale=(0.5))

    plt.plot(x, rayleigh_pdf, 'r-', lw=2, label='Rayleigh Distribution PDF (scale=0.5)')

    plt.title(title)
    plt.xlabel('Pixel Values')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.show()

def load_images_from_folder(folder_path):
    image_dict = {}
    for filename in os.listdir(folder_path):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(folder_path, filename)
            img = io.read_image(img_path)
            
            # Ensure images are loaded as grayscale
            if img.shape[0] == 3:  # If the image is RGB, convert it to grayscale
                img = transforms.Grayscale()(img)

            filename = filename[0:7]
            image_dict[filename] = img
            
    return image_dict

def main():
    real_A = load_images_from_folder('gray_ray_real_A')
    fake_B = load_images_from_folder('gray_ray_fake_B')

    noises = []
    filters = []
    for filename, a in real_A.items():
        b = fake_B[filename]

        # Ensure both tensors have the same number of channels
        if a.shape[0] != b.shape[0]:
            if a.shape[0] == 3:
                a = transforms.Grayscale()(a)
            if b.shape[0] == 3:
                b = transforms.Grayscale()(b)

        valid_indices = (a != 0) & (b != 0)
        noise = b[valid_indices] / a[valid_indices]
        filtered = noise[noise < 2]
        noises.append(noise)
        filters.append(filtered)

    print("processed images")

    # file_path = 'noise.pkl'

    # # Open the pickle file and load the tensor
    # with open(file_path, 'rb') as f:
    #     tensor = pickle.load(f)

    # print("loaded pickle")

    flat_data = torch.cat(noises)
    flat_data_filter = torch.cat(filters)
    # plot_pdf_with_rayleigh(tensor, 'Original Noise')
    plot_pdf_with_rayleigh(flat_data, 'Noise / Normal for CycleGAN')
    plot_pdf_with_rayleigh(flat_data_filter, 'Noise / Normal for CycleGAN [0-2]')
    # ks_statistic, p_value = kstest(flat_data.numpy(), tensor.numpy())
    # print("processed images")
    # print(f'K-S Test Statistic: {ks_statistic}, P-Value: {p_value}')

if __name__ == "__main__":
    main()
