import sys
sys.path.append("/blue/azare/samgallic/Research/new_cycle_gan")
import torch
import copy
from data.unaligned_dataset import UnalignedDataset
from PIL import Image
import os

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

        k = 256*256
        indices_gam = torch.randperm(self.emp_gamma.size(0))[:k]
        indices_ray = torch.randperm(self.emp_rayleigh.size(0))[:k]

        self.emp_gamma = self.emp_gamma[indices_gam]
        self.emp_rayleigh = self.emp_rayleigh[indices_ray]
        
        # Debug: Check tensor dimensions
        print(f"emp_gamma shape: {self.emp_gamma.shape}")
        print(f"emp_rayleigh shape: {self.emp_rayleigh.shape}")

    def earth_movers(self, model):
        noise_gam_total = 0.0
        noise_ray_total = 0.0
        
        for path_A, path_B, a, b in zip(model.paths['A'], model.paths['B'], model.fake_A, model.fake_B):
            path_A = os.path.basename(path_A)
            path_B = os.path.basename(path_B)

            # Compute noise for the generated images
            noise_ray = (b - self.unnoise_A[path_A]).view(-1).float()
            noise_gam = (self.unnoise_B[path_B] - a).view(-1).float()

            wd_ray = self.loss(noise_ray, self.emp_rayleigh)
            wd_gam = self.loss(noise_gam, self.emp_gamma)

            noise_ray_total += wd_ray
            noise_gam_total += wd_gam

        # Average out the distances over the number of paths (batch size)
        num_paths = len(model.paths['A'])
        if num_paths == 0:
            print("No paths found in model.paths['A'].")
            return 0.0  # Avoid division by zero
        noisy_gam_avg = (noise_gam_total / num_paths).item()
        noisy_ray_avg = (noise_ray_total / num_paths).item()

        return noisy_ray_avg, noisy_gam_avg
    
    def loss(self, sample, empirical):
        # Determine the min and max from both tensors
        data_min = min(sample.min(), empirical.min())
        data_max = max(sample.max(), empirical.max())

        # Define the bins based on the min and max values
        bins = torch.linspace(data_min.item(), data_max.item(), steps=500)

        # Calculate the histograms for each tensor
        hist1 = torch.histc(sample, bins=len(bins), min=bins.min().item(), max=bins.max().item())
        hist2 = torch.histc(empirical, bins=len(bins), min=bins.min().item(), max=bins.max().item())

        # Subtract the histograms element-wise
        hist_diff = torch.abs(hist1 - hist2).mean()
        return hist_diff