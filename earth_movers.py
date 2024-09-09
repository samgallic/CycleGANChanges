import histogram
import torch
from scipy.stats import wasserstein_distance
from data.unaligned_dataset import UnalignedDataset
import shutil
import os
from torchvision import transforms

class DistanceCalc:
    def __init__(self, opt):
        self.opt = opt
        self.opt.no_flip = True
        self.opt.serial_batches = True
        self.dataset = UnalignedDataset(self.opt)
        real_A = histogram.load_images_from_folder(self.opt.dataroot + '/trainA')
        noisy_A = histogram.add_multiplicative_rayleigh_noise(real_A, 0.5)
        os.makedirs('temp', exist_ok=True)
        for filename, img in noisy_A.items():
            histogram.save_png(img, 'temp/' + filename)
        emp_noisy_A = histogram.load_images_from_folder('temp')
        shutil.rmtree('temp')
        
        noises = []
        for filename, a in real_A.items():
            b = emp_noisy_A[filename]
            if a.shape[0] != b.shape[0]:
                if a.shape[0] == 3:
                    a = transforms.Grayscale()(a)
                if b.shape[0] == 3:
                    b = transforms.Grayscale()(b)
            valid_indices = (a != 0) & (b != 0)
            noise = b[valid_indices] / a[valid_indices]
            noises.append(noise)
        self.emp = torch.cat(noises)

    def earth_movers(self, model):
        model.netG_A.eval()
        model.netG_B.eval()
        with torch.no_grad():
            real_A = {}
            real_B = {}
            fake_A = {}
            fake_B = {}
            for i in range(len(self.dataset)):
                data = self.dataset[i]
                A_images = data['A'].to(model.device)
                B_images = data['B'].to(model.device)
                A_paths = data['A_paths']
                B_paths = data['B_paths']
                real_A[A_paths] = A_images.to('cpu')
                real_B[B_paths] = B_images.to('cpu')
                fake_B[A_paths] = model.netG_A(A_images).to('cpu')
                fake_A[B_paths] = model.netG_B(B_images).to('cpu')
        model.netG_A.train()
        model.netG_B.train()

        noises = []
        for filename, a in real_A.items():
            b = fake_B[filename]
            if a.shape[0] != b.shape[0]:
                if a.shape[0] == 3:
                    a = transforms.Grayscale()(a)
                if b.shape[0] == 3:
                    b = transforms.Grayscale()(b)
            valid_indices = (a != 0) & (b != 0)
            noise = b[valid_indices] / a[valid_indices]
            noises.append(noise)
        noisy_a = torch.cat(noises)

        noises = []
        for filename, a in fake_A.items():
            b = real_B[filename]
            if a.shape[0] != b.shape[0]:
                if a.shape[0] == 3:
                    a = transforms.Grayscale()(a)
                if b.shape[0] == 3:
                    b = transforms.Grayscale()(b)
            valid_indices = (a != 0) & (b != 0)
            noise = b[valid_indices] / a[valid_indices]
            noises.append(noise)
        noisy_b = torch.cat(noises)

        emd_A = wasserstein_distance(self.emp, noisy_a)
        emd_B = wasserstein_distance(self.emp, noisy_b)

        dists = {}
        dists['emd_A'] = emd_A
        dists['emd_B'] = emd_B
        return dists