import torch
from models.networks import NoiseGenerator
from data.unaligned_dataset import UnalignedDataset
from PIL import Image
import os
import copy
from util.util import tensor2im

def load_images_from_folder(folder_path):
    image_dict = {}
    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(folder_path, filename)
            img = Image.open(img_path).convert('RGB')  # Ensure images are in RGB
            image_dict[filename] = img
    return image_dict
    
def load_generators(model_opt_name):
    # Instantiate two NoiseGenerator models
    generator_A = NoiseGenerator()
    generator_B = NoiseGenerator()

    # Define paths for the state dictionaries
    state_dict_A_path = f'/blue/azare/samgallic/Research/new_cycle_gan/checkpoints/{model_opt_name}/latest_net_G_A.pth'
    state_dict_B_path = f'/blue/azare/samgallic/Research/new_cycle_gan/checkpoints/{model_opt_name}/latest_net_G_B.pth'
    
    # Load the state dictionaries
    generator_A.load_state_dict(torch.load(state_dict_A_path))
    generator_B.load_state_dict(torch.load(state_dict_B_path))
    
    return generator_A, generator_B

def get_intermediate_output(model, layer_index, input_tensor):
    intermediate_output = None
    
    # Define the hook function
    def hook(module, input, output):
        nonlocal intermediate_output
        intermediate_output = output

    # Register the hook on the desired layer
    handle = model.module.model[layer_index].register_forward_hook(hook)
    
    # Run the forward pass
    _ = model(input_tensor)
    
    # Remove the hook
    handle.remove()
    
    return intermediate_output

class CleanLoss():
    def __init__(self, model):
        # Create a deep copy of the opt object to avoid modifying the original
        self.opt = copy.deepcopy(model.opt)
        self.device = model.device
        
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
        
        # Process images and compute noises
        for filename, img in real_A_pil.items():
            if filename not in unnoise_A_pil:
                print(f"Warning: {filename} not found in unnoise_A_pil")
                continue
            normal = transform_A(unnoise_A_pil[filename])
            gamma = transform_A(img)
            self.real_A[filename] = gamma.to(model.device)
            self.unnoise_A[filename] = normal.to(model.device)
        
        for filename, img in real_B_pil.items():
            if filename not in unnoise_B_pil:
                print(f"Warning: {filename} not found in unnoise_B_pil")
                continue
            normal = transform_B(unnoise_B_pil[filename])
            rayleigh = transform_B(img)
            self.real_B[filename] = rayleigh.to(model.device)
            self.unnoise_B[filename] = normal.to(model.device)

    def clean_loss(self, opt, paths_A, paths_B, model):
        # Replace 'your_model_name' with the actual name you want to use
        generator_A = copy.deepcopy(model.netG_A)
        generator_B = copy.deepcopy(model.netG_B)
        generator_A.to(model.device)
        generator_B.to(model.device)
        clean_A = {}
        clean_B = {}
        loss_A = 0.0
        loss_B = 0.0
        os.makedirs('clean', exist_ok=True)
        for path_A, path_B in zip(paths_A, paths_B):
            path_A = os.path.basename(path_A)
            path_B = os.path.basename(path_B)
            clean_A[path_A] = get_intermediate_output(generator_A, 12, self.real_A[path_A])
            Image.fromarray(tensor2im(clean_A[path_A])).save(f'clean/{path_A}')
            clean_B[path_B] = get_intermediate_output(generator_B, 12, self.real_B[path_B])
            Image.fromarray(tensor2im(clean_B[path_B])).save(f'clean/{path_B}')
            loss_A = torch.abs(clean_A[path_A] - self.unnoise_A[path_A]).mean()
            loss_B = torch.abs(clean_B[path_B] - self.unnoise_B[path_B]).mean()
        return loss_A / len(path_A), loss_B / len(path_B)
