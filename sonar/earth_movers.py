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
        unnoise_B_pil = {}

        if self.opt.isTrain:
            real_A_pil = load_images_from_folder(opt.dataroot + '/trainA')
            real_B_pil = load_images_from_folder(opt.dataroot + '/trainB')
            unnoise_B_pil = load_images_from_folder(new_opt.dataroot + '/trainB')
        else:
            real_A_pil = load_images_from_folder(opt.dataroot + '/testA')
            real_B_pil = load_images_from_folder(opt.dataroot + '/testB')
            unnoise_B_pil = load_images_from_folder(new_opt.dataroot + '/testB')

        self.real_A = {}
        self.real_B = {}
        noises = []
        for filename, img in real_A_pil.items():
            self.real_A[filename] = transform_A(img).to(model.device)
        for filename, img in real_B_pil.items():
            a = transform_B(unnoise_B_pil[filename])
            b = transform_B(img)
            self.real_B[filename] = b.to(model.device)
            noise = b - a
            noises.append(noise)
        self.emp = torch.cat(noises)

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
        noisy_a = self._calculate_noises_batch(self.real_A, fake_B)
        noisy_b = self._calculate_noises_batch(fake_A, self.real_B)
        self.emp = self.emp.cpu()
        noisy_a = noisy_a.cpu()
        noisy_b = noisy_b.cpu()

        # Split tensors into batches
        emp_batches = batch_tensor(self.emp, batch_size)
        noisy_a_batches = batch_tensor(noisy_a, batch_size)
        noisy_b_batches = batch_tensor(noisy_b, batch_size)

        # Initialize variables to store EMD for each batch
        emd_A_total = 0
        emd_B_total = 0

        # Compute EMD for each batch and accumulate results
        for emp_batch, noisy_a_batch in zip(emp_batches, noisy_a_batches):
            emd_A_total += wasserstein_distance(emp_batch.flatten(), noisy_a_batch.flatten())
            
        for emp_batch, noisy_b_batch in zip(emp_batches, noisy_b_batches):
            emd_B_total += wasserstein_distance(emp_batch.flatten(), noisy_b_batch.flatten())

        emd_A_avg = emd_A_total / len(emp_batches)
        emd_B_avg = emd_B_total / len(emp_batches)

        ks_stat_A = ks_2samp(self.emp.flatten().numpy(), noisy_a.flatten().numpy()).statistic
        ks_stat_B = ks_2samp(self.emp.flatten().numpy(), noisy_b.flatten().numpy()).statistic
        print(ks_stat_A)
        print(ks_stat_B)

        if epoch % 50 == 0 or epoch == 1:
            path = ''
            if self.opt.isTrain:
                path = f'/blue/azare/samgallic/Research/new_cycle_gan/checkpoints/{self.opt.name}/histograms'
            else:
                path = f'/blue/azare/samgallic/Research/new_cycle_gan/results/{self.opt.name}/histograms'
            os.makedirs(path, exist_ok=True)
            histogram.plot_pdf_with_rayleigh(noisy_a.cpu(), self.emp.cpu(), f'Epoch {epoch} Normal2Noisy', f'{path}/{epoch}_A.png')
            histogram.plot_pdf_with_rayleigh(noisy_b.cpu(), self.emp.cpu(), f'Epoch {epoch} Noisy2Normal', f'{path}/{epoch}_B.png')

        return {'emd_A': emd_A_avg, 'emd_B': emd_B_avg, 'ks_A': ks_stat_A, 'ks_B': ks_stat_B}


    def _calculate_noises_batch(self, A_batch, B_batch):
        noises = []
        for filename, a in A_batch.items():
            b = B_batch[filename]
            a, b = self._shape_check(a, b)
            noise = b - a
            noises.append(noise)
        return torch.cat(noises)
