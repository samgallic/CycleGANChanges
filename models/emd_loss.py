import sys
sys.path.append("/blue/azare/samgallic/Research/new_cycle_gan")
import torch
import copy
from data.unaligned_dataset import UnalignedDataset
from PIL import Image
import os
import kornia

def debug_tensor(tensor, name):
    print(f"Debugging tensor {name}:")
    print(f"Mean: {tensor.mean().item()}, Min: {tensor.min().item()}, Max: {tensor.max().item()}, Std: {tensor.std().item()}")

def load_images_from_folder(folder_path):
    image_dict = {}
    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(folder_path, filename)
            img = Image.open(img_path).convert('RGB')  # Ensure images are in RGB
            image_dict[filename] = img
    return image_dict

class DistanceCalc:
    def __init__(self, model):
        # Create a deep copy of the opt object to avoid modifying the original
        self.opt = copy.deepcopy(model.opt)
        
        # Ensure no changes to the original opt
        self.opt.no_flip = True
        self.opt.serial_batches = True
        self.opt.batch_size = 1

        self.dataset = UnalignedDataset(self.opt)
        new_opt = copy.deepcopy(self.opt)
        new_opt.dataroot = '/blue/azare/samgallic/Research/new_cycle_gan/datasets/gray_forest'
        transform_A = self.dataset.transform_A
        transform_B = self.dataset.transform_B

        # Load images from directories
        real_A_pil = load_images_from_folder(os.path.join(model.opt.dataroot, 'trainA'))
        real_B_pil = load_images_from_folder(os.path.join(model.opt.dataroot, 'trainB'))
        unnoise_A_pil = load_images_from_folder(os.path.join(new_opt.dataroot, 'trainA'))
        unnoise_B_pil = load_images_from_folder(os.path.join(new_opt.dataroot, 'trainB'))

        self.real_A = {}
        self.real_B = {}
        self.unnoise_A = {}
        self.unnoise_B = {}
        noises_A = []
        noises_B = []
        
        # Process images and compute noises
        for filename, img in real_A_pil.items():
            if filename not in unnoise_A_pil:
                print(f"Warning: {filename} not found in unnoise_A_pil")
                continue
            normal = transform_A(unnoise_A_pil[filename])
            gamma = transform_A(img)
            self.real_A[filename] = gamma.to(model.device)
            self.unnoise_A[filename] = normal.to(model.device)
            noise = gamma - normal
            noises_A.append(noise)
        
        for filename, img in real_B_pil.items():
            if filename not in unnoise_B_pil:
                print(f"Warning: {filename} not found in unnoise_B_pil")
                continue
            normal = transform_B(unnoise_B_pil[filename])
            rayleigh = transform_B(img)
            self.real_B[filename] = rayleigh.to(model.device)
            self.unnoise_B[filename] = normal.to(model.device)
            noise = rayleigh - normal
            noises_B.append(noise)
        
        # Store the entire distribution of noises as 1D tensors
        self.emp_gamma = torch.cat(noises_A).view(-1).float().to(model.device)
        self.emp_rayleigh = torch.cat(noises_B).view(-1).float().to(model.device)

        indices = torch.randperm(self.emp_gamma.size(0))[:(256*256)]
        self.emp_gamma = self.emp_gamma[indices]
        self.emp_rayleigh = self.emp_rayleigh[indices]

        self.num_bins = 256
        self.bandwidth = torch.tensor(0.1, device=model.device)
        
        self.sanity_path = "/blue/azare/samgallic/Research/new_cycle_gan/checkpoints/" + self.opt.name + "/sanity_check"
        os.makedirs(self.sanity_path, exist_ok=True)

        self.compute_bins()
        self.compute_empirical_histograms()

    def compute_bins(self):
        # Compute the min and max for bin edges
        self.emp_gamma_min = self.emp_gamma.min()
        self.emp_gamma_max = self.emp_gamma.max()
        self.emp_rayleigh_min = self.emp_rayleigh.min()
        self.emp_rayleigh_max = self.emp_rayleigh.max()

        # Create bins for gamma and rayleigh without unsqueeze(0)
        self.bins_gamma = torch.linspace(
            self.emp_gamma_min, self.emp_gamma_max, steps=self.num_bins, device=self.emp_gamma.device
        )  # Shape: (num_bins,)

        self.bins_rayleigh = torch.linspace(
            self.emp_rayleigh_min, self.emp_rayleigh_max, steps=self.num_bins, device=self.emp_rayleigh.device
        )  # Shape: (num_bins,)

    def compute_empirical_histograms(self):
        self.emp_gamma_hist = self.compute_histogram(self.emp_gamma, self.bins_gamma)
        self.emp_rayleigh_hist = self.compute_histogram(self.emp_rayleigh, self.bins_rayleigh)

    def compute_histogram(self, data, bins):
        data = data.unsqueeze(0)  # Shape: (1, N)
        hist = kornia.enhance.histogram(data, bins, self.bandwidth)
        hist = hist / (hist.sum() + 1e-10)  # Normalize histogram
        return hist  # Shape: (1, num_bins)

    def earth_movers(self, model):
        noise_gam_total = 0.0
        noise_ray_total = 0.0

        for path_A, path_B in zip(model.paths['A'], model.paths['B']):
            path_A = os.path.basename(path_A)
            path_B = os.path.basename(path_B)

            b = model.netG_A(self.real_A[path_A])
            a = model.netG_B(self.real_B[path_B])

            # Compute noise for the generated images
            noise_ray = (b - self.unnoise_A[path_A]).view(-1)
            noise_gam = (self.unnoise_B[path_B] - a).view(-1)

            # Compute histograms
            hist_noise_ray = self.compute_histogram(noise_ray, self.bins_rayleigh)
            hist_noise_gam = self.compute_histogram(noise_gam, self.bins_gamma)

            # Normalize histograms
            hist_noise_ray = hist_noise_ray / (hist_noise_ray.sum() + 1e-10)
            hist_noise_gam = hist_noise_gam / (hist_noise_gam.sum() + 1e-10)

            # Compute loss between histograms
            wd_ray = torch.nn.functional.l1_loss(hist_noise_ray, self.emp_rayleigh_hist)
            wd_gam = torch.nn.functional.l1_loss(hist_noise_gam, self.emp_gamma_hist)

            noise_ray_total += wd_ray
            noise_gam_total += wd_gam

        num_paths = len(model.paths['A'])
        if num_paths == 0:
            print("No paths found in model.paths['A'].")
            return 0.0, 0.0

        noisy_gam_avg = noise_gam_total / num_paths
        noisy_ray_avg = noise_ray_total / num_paths

        return noisy_ray_avg, noisy_gam_avg
