import sys
sys.path.append("/blue/azare/samgallic/Research/new_cycle_gan")
import torch
import copy
from data.unaligned_dataset import UnalignedDataset
from PIL import Image
import os
import matplotlib.pyplot as plt
from util.util import tensor2im, save_image

def loss(sample, empirical):
    # Ensure tensors are on the same device (i.e., CUDA if available)
    assert sample.device == empirical.device, "Tensors should be on the same device."

    # Ensure that the tensors do not contain NaN or Inf values
    if torch.isnan(sample).any() or torch.isnan(empirical).any():
        raise ValueError("Input tensors contain NaN values")
    if torch.isinf(sample).any() or torch.isinf(empirical).any():
        raise ValueError("Input tensors contain Inf values")

    # Determine the min and max from both tensors
    data_min = min(sample.min(), empirical.min())
    data_max = max(sample.max(), empirical.max())

    # Define the bins based on the min and max values
    bins = torch.linspace(data_min.item(), data_max.item(), steps=500, device=sample.device)

    # Compute histograms (make sure they are on the same device)
    hist1 = torch.histc(sample, bins=len(bins), min=bins.min().item(), max=bins.max().item())
    hist2 = torch.histc(empirical, bins=len(bins), min=bins.min().item(), max=bins.max().item())

    cdf_1 = torch.cumsum(hist1, dim=0)
    cdf_2 = torch.cumsum(hist2, dim=0)

    cdf_1 /= cdf_1.clone()[-1]
    cdf_2 /= cdf_2.clone()[-1]

    # Subtract the histograms element-wise
    cdf_diff = torch.abs(cdf_1 - cdf_2).sum()

    return cdf_diff.item()  # Ensure the return type is a float

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
        
        self.sanity_path = "/blue/azare/samgallic/Research/new_cycle_gan/checkpoints/" + self.opt.name + "/sanity_check"
        os.makedirs(self.sanity_path, exist_ok=True)

    def earth_movers(self, model):
        noise_gam_total = 0.0
        noise_ray_total = 0.0

        for path_A, path_B, in zip(model.paths['A'], model.paths['B']):
            path_A = os.path.basename(path_A)
            path_B = os.path.basename(path_B)

            b = model.netG_A(self.real_A[path_A])
            a = model.netG_B(self.real_B[path_B])

            # Compute noise for the generated images
            noise_ray = (b - self.unnoise_A[path_A]).view(-1).float()
            noise_gam = (self.unnoise_B[path_B] - a).view(-1).float()

            wd_ray = loss(noise_ray, self.emp_rayleigh)
            wd_gam = loss(noise_gam, self.emp_gamma)

            noise_ray_total += wd_ray
            noise_gam_total += wd_gam

        # Average out the distances over the number of paths (batch size)
        num_paths = len(model.paths['A'])
        if num_paths == 0:
            print("No paths found in model.paths['A'].")
            return 0.0  # Avoid division by zero
        noisy_gam_avg = noise_gam_total / num_paths
        noisy_ray_avg = noise_ray_total / num_paths

        noisy_ray_avg = torch.tensor(noisy_ray_avg, requires_grad=True)
        noisy_gam_avg = torch.tensor(noisy_gam_avg, requires_grad=True)

        return noisy_ray_avg, noisy_gam_avg
