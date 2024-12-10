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

def save_png(tensor, filename):
    # Ensure the filename has a valid extension
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        filename += '.png'  # Default to .png if no valid extension is provided

    # Create directories if they don't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    tensor = torch.div(tensor, 255.0)
    
    # Save the image
    torchvision.utils.save_image(tensor, filename)

def add_multiplicative_rayleigh_noise(images, scale):
    noisy_images = {}

    for filename, img in images.items():
        # Generate Rayleigh noise using numpy
        rayleigh_noise = np.random.rayleigh(scale, size=img.shape)

        # Convert the numpy array to a PyTorch tensor and move it to the same device as the image tensor
        rayleigh_noise_tensor = torch.from_numpy(rayleigh_noise).to(img.device)

        # Apply multiplicative noise
        noisy_img = torch.mul(img, rayleigh_noise_tensor)

        # Add the noisy image to the dictionary with the same filename as the key
        noisy_images[filename] = noisy_img

    return noisy_images

def plot_pdf_with_rayleigh(data, data_emp, title, save_path):
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
    
    # Flatten and prepare data_emp
    if isinstance(data_emp, np.ndarray):
        flat_data_emp = data_emp.flatten()
    elif isinstance(data_emp, list) and isinstance(data_emp[0], np.ndarray):
        flat_data_emp = np.concatenate([d.flatten() for d in data_emp])
    else:
        flat_data_emp = torch.cat([d.flatten() for d in data_emp])
        flat_data_emp = flat_data_emp[torch.isfinite(flat_data_emp)]

    print(f"Noise Values - Min: {flat_data.min()}, Max: {flat_data.max()}, Mean: {flat_data.mean()}")

    plt.figure(figsize=(10, 6))
    
    # Plot the empirical data in faded color
    plt.hist(flat_data_emp, bins=200, color='gray', alpha=0.3, density=True, label='Empirical Data')
    
    # Plot the main data
    plt.hist(flat_data, bins=200, color='blue', alpha=0.7, density=True, label='Noise Data')

    plt.title(title)
    plt.xlabel('Pixel Values')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.xlim([-2, 2])
    
    # Save the plot as a PNG file
    plt.savefig(save_path, format='png', bbox_inches='tight')

def load_images_from_folder(folder_path):
    image_dict = {}
    for filename in os.listdir(folder_path):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(folder_path, filename)
            img = io.read_image(img_path)

            image_dict[filename] = img
            
    return image_dict

def main():
    name = 'n2n_16_timed'
    output = 'noisy'

    real_A = load_images_from_folder('results/organized/'+ name + '/fake_A')
    fake_B = load_images_from_folder('results/organized/'+ name + '/real_B')
    noises_emp = []

    real_noise = add_multiplicative_rayleigh_noise(real_A, 0.5)
    
    output = 'noisy'
    os.makedirs(output, exist_ok=True)
    for filename, b in real_noise.items():
        b = real_noise[filename]
        save_png(b, output + '/' + filename)

    real_noise = load_images_from_folder(output)

    noises = []
    filters = []
    for filename, a in real_A.items():
        b = fake_B[filename]
        b_emp = real_noise[filename]

        # Ensure both tensors have the same number of channels
        if a.shape[0] != b.shape[0]:
            if a.shape[0] == 3:
                a = transforms.Grayscale()(a)
            if b.shape[0] == 3:
                b = transforms.Grayscale()(b)

        if a.shape[0] != b_emp.shape[0]:
            if a.shape[0] == 3:
                a = transforms.Grayscale()(a)
            if b.shape[0] == 3:
                b_emp = transforms.Grayscale()(b_emp)

        valid_indices = (a != 0) & (b != 0)
        noise = b[valid_indices] / a[valid_indices]
        noises.append(noise)

        filtered = noise[noise < 2]
        filters.append(filtered)

        valid_indices_emp = (a != 0) & (b_emp != 0)
        noise_emp = b_emp[valid_indices_emp] / a[valid_indices_emp]
        noises_emp.append(noise_emp)

    print("processed images")

    flat_data = torch.cat(noises)
    emp_flat = torch.cat(noises_emp)
    flat_data_filter = torch.cat(filters)
    plot_pdf_with_rayleigh(flat_data, emp_flat, 'Noise / Normal for CycleGAN')
    plot_pdf_with_rayleigh(flat_data_filter, emp_flat, 'Noise / Normal for CycleGAN [0-2]')

if __name__ == "__main__":
    main()
