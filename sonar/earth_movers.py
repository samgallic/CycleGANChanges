import sys
sys.path.append("/blue/azare/samgallic/Research/new_cycle_gan")
import torch
import copy
from scipy.stats import wasserstein_distance
from data.unaligned_dataset import UnalignedDataset
from torchvision import transforms
import sonar.histogram as histogram
from scipy.stats import ks_2samp
from PIL import Image
import os

def batch_tensor(tensor, batch_size):
    # Split the tensor into smaller batches
    return torch.split(tensor, batch_size)

def filter_tensor(tensor, min_value, max_value):
    # Create a mask for values within the desired range [min_value, max_value]
    mask = (tensor >= min_value) & (tensor <= max_value)
    
    # Apply the mask to filter the tensor
    filtered_tensor = tensor[mask]
    
    return filtered_tensor

def load_images_from_folder(folder_path):
    image_dict = {}
    for filename in os.listdir(folder_path):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(folder_path, filename)
            img = Image.open(img_path)
            image_dict[filename] = img
            
    return image_dict

class DistanceCalc:
    def __init__(self, opt, model):
        # Create a deep copy of the opt object to avoid modifying the original
        self.opt = copy.deepcopy(opt)
        
        # Ensure no changes to the original opt
        self.opt.no_flip = True
        self.opt.serial_batches = True
        self.opt.batch_size = 1

        self.dataset = UnalignedDataset(self.opt)
        new_opt = copy.deepcopy(self.opt)
        new_opt.dataroot = '/blue/azare/samgallic/Research/new_cycle_gan/datasets/gray_forest'
        transform_A = self.dataset.transform_A
        transform_B = self.dataset.transform_B

        real_A_pil = {}
        real_B_pil = {}
        unnoise_A_pil = {}
        unnoise_B_pil = {}

        if self.opt.isTrain:
            real_A_pil = load_images_from_folder(opt.dataroot + '/trainA')
            real_B_pil = load_images_from_folder(opt.dataroot + '/trainB')
            unnoise_A_pil = load_images_from_folder(new_opt.dataroot + '/trainA')
            unnoise_B_pil = load_images_from_folder(new_opt.dataroot + '/trainB')
        else:
            real_A_pil = load_images_from_folder(opt.dataroot + '/testA')
            real_B_pil = load_images_from_folder(opt.dataroot + '/testB')
            unnoise_A_pil = load_images_from_folder(new_opt.dataroot + '/testA')
            unnoise_B_pil = load_images_from_folder(new_opt.dataroot + '/testB')

        self.real_A = {}
        self.real_B = {}
        self.unnoise_A = {}
        self.unnoise_B = {}
        noises_A = []
        noises_B = []
        for filename, img in real_A_pil.items():
            normal = transform_A(unnoise_A_pil[filename])
            gamma = transform_A(img)
            self.real_A[filename] = gamma.to(model.device)
            self.unnoise_A[filename] = normal.to(model.device)
            noise = gamma - normal
            noises_A.append(noise)
        for filename, img in real_B_pil.items():
            normal = transform_B(unnoise_B_pil[filename])
            rayleigh = transform_B(img)
            self.real_B[filename] = rayleigh.to(model.device)
            self.unnoise_B[filename] = normal.to(model.device)
            noise = rayleigh - normal
            noises_B.append(noise)
        self.emp_gamma = torch.cat(noises_A)
        self.emp_rayleigh = torch.cat(noises_B)

    def _shape_check(self, a, b):
        if a.shape[0] != 1 or b.shape[0] != 1:
            if a.shape[0] == 3:
                a = transforms.Grayscale()(a)
            if b.shape[0] == 3:
                b = transforms.Grayscale()(b)
        return a, b

    def earth_movers(self, epoch, model, batch_size=50):
        model.netG_A.eval()
        model.netG_B.eval()

        fake_A, fake_B = {}, {}

        with torch.no_grad():
            for filename in self.real_A.keys():
                fake_B[filename] = model.netG_A(self.real_A[filename].to(model.device))
            for filename in self.real_B.keys():
                fake_A[filename] = model.netG_B(self.real_B[filename].to(model.device))

        model.netG_A.train()
        model.netG_B.train()

        # Calculate noises in batches for the fake images
        noisy_gamma = self._calculate_noises_batch(self.unnoise_A, fake_B)
        noisy_rayleigh = self._calculate_noises_batch(fake_A, self.unnoise_B)
        self.emp_rayleigh = self.emp_rayleigh.cpu()
        self.emp_gamma = self.emp_gamma.cpu()
        noisy_gamma = noisy_gamma.cpu()
        noisy_rayleigh = noisy_rayleigh.cpu()

        # Split tensors into batches
        emp_gamma_batches = batch_tensor(self.emp_gamma, batch_size)
        emp_rayleigh_batches = batch_tensor(self.emp_rayleigh, batch_size)
        noisy_gamma_batches = batch_tensor(noisy_gamma, batch_size)
        noisy_rayleigh_batches = batch_tensor(noisy_rayleigh, batch_size)

        # Initialize variables to store EMD for each batch
        emd_A_total = 0
        emd_B_total = 0

        # Compute EMD for each batch and accumulate results
        for emp_gamma_batch, noisy_gamma_batch in zip(emp_gamma_batches, noisy_gamma_batches):
            emd_A_total += wasserstein_distance(emp_gamma_batch.flatten(), noisy_gamma_batch.flatten())
            
        for emp_batch, noisy_rayleigh_batch in zip(emp_rayleigh_batches, noisy_rayleigh_batches):
            emd_B_total += wasserstein_distance(emp_batch.flatten(), noisy_rayleigh_batch.flatten())

        emd_A_avg = emd_A_total / len(noisy_gamma_batches)
        emd_B_avg = emd_B_total / len(emp_rayleigh_batches)

        ks_stat_A = ks_2samp(self.emp_gamma.flatten().numpy(), noisy_gamma.flatten().numpy()).statistic
        ks_stat_B = ks_2samp(self.emp_rayleigh.flatten().numpy(), noisy_rayleigh.flatten().numpy()).statistic

        if epoch % 50 == 0 or epoch == 1:
            path = ''
            if self.opt.isTrain:
                path = f'/blue/azare/samgallic/Research/new_cycle_gan/checkpoints/{self.opt.name}/histograms'
                os.makedirs(path, exist_ok=True)
                histogram.plot_pdf_with_rayleigh(noisy_gamma.cpu(), self.emp_gamma.cpu(), f'Epoch {epoch} Rayleigh2Gamma', f'{path}/Epoch_{epoch}_Gamma.png')
                histogram.plot_pdf_with_rayleigh(noisy_rayleigh.cpu(), self.emp_rayleigh.cpu(), f'Epoch {epoch} Gamma2Rayleigh', f'{path}/Epoch_{epoch}_Rayleigh.png')
            else:
                path = f'/blue/azare/samgallic/Research/new_cycle_gan/results/{self.opt.name}/histograms'
                os.makedirs(path, exist_ok=True)
                histogram.plot_pdf_with_rayleigh(noisy_rayleigh.cpu(), self.emp_rayleigh.cpu(), f'Test Gamma2Rayleigh', f'{path}/test_Rayleigh.png')
                histogram.plot_pdf_with_rayleigh(noisy_gamma.cpu(), self.emp_gamma.cpu(), f'Test Rayleigh2Gamma', f'{path}/test_Gamma.png')

        return {'emd_A': emd_A_avg, 'emd_B': emd_B_avg, 'ks_A': ks_stat_A, 'ks_B': ks_stat_B}


    def _calculate_noises_batch(self, A_batch, B_batch):
        noises = []
        for filename, a in A_batch.items():
            b = B_batch[filename]
            a, b = self._shape_check(a, b)
            noise = b - a
            noises.append(noise)
        return torch.cat(noises)
