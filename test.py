"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""

import os
import torch
import numpy as np
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
from torchmetrics.image import StructuralSimilarityIndexMeasure
import matplotlib.pyplot as plt
from torch import nn
from torch.nn import functional as F
import torch.utils.data
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from scipy.stats import entropy
from torchvision.models import inception_v3
from cleanfid import fid
from torchvision.transforms import functional as TF
import csv

def bootstrap_fid(real_images, fake_images, n_bootstrap=100):
    # Check if GPU is available and set the device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # Move the images to the specified device (GPU if available)
    real_images = real_images.to(device)
    fake_images = fake_images.to(device)

    n_samples = real_images.shape[0]
    fid_scores = []

    for i in range(n_bootstrap):
        # Sample indices with replacement
        sampled_indices = np.random.choice(n_samples, n_samples, replace=True)
        print(f"{i+1} / {n_bootstrap}")

        # Resample the images
        real_resampled = real_images[sampled_indices].to(device)
        fake_resampled = fake_images[sampled_indices].to(device)
        
        # Initialize FID object (does not need the device parameter)
        fid = FrechetInceptionDistance(feature=2048)
        fid = fid.to(device)  # Move the FID object to the correct device manually
        fid.update(real_resampled, real=True)
        fid.update(fake_resampled, real=False)
        
        # Compute FID for the resampled set
        fid_score = fid.compute()
        fid_scores.append(fid_score.item())  # Convert tensor to Python scalar

    return fid_scores

try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')

if __name__ == '__main__':

    # Paths to your image folders
    folder_fake = 'monet2photo_fake_B'  # Update this path
    folder_real = 'monet2photo_real_B'  # Update this path

    # Compute the FID score
    # score = fid.compute_fid(folder_fake, folder_real)

    # print(score)

    fake_A_images = []
    fake_B_images = []
    real_A_images = []
    real_B_images = []
    cycle_images = []
    inception = InceptionScore()

    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    # initialize logger
    if opt.use_wandb:
        wandb_run = wandb.init(project=opt.wandb_project_name, name=opt.name, config=opt) if not wandb.run else wandb.run
        wandb_run._label(repo='CycleGAN-and-pix2pix')

    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))

    scores = []
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        fake_A_images.append(model.fake_A)
        fake_B_images.append(model.fake_B)
        real_A_images.append(model.real_A)
        real_B_images.append(model.real_B)
        cycle_images.append(model.rec_A)
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, use_wandb=opt.use_wandb)
    webpage.save()  # save the HTML

    real_A_images_tensor = ((torch.cat(real_A_images, dim=0) + 1) * 0.5 * 255).byte().cpu()
    fake_A_images_tensor = ((torch.cat(fake_A_images, dim=0) + 1) * 0.5 * 255).byte().cpu()
    real_B_images_tensor = ((torch.cat(real_B_images, dim=0) + 1) * 0.5 * 255).byte().cpu()
    fake_B_images_tensor = ((torch.cat(fake_B_images, dim=0) + 1) * 0.5 * 255).byte().cpu()

    fid_scores = bootstrap_fid(real_B_images_tensor, fake_B_images_tensor)
    print("Mean FID:", np.mean(fid_scores))
    print("FID Std Dev:", np.std(fid_scores))

    inception.update(fake_B_images_tensor)
    print("Inception Score: ", inception.compute())

    fid = FrechetInceptionDistance(feature=2048)
    
    fid.update(real_B_images_tensor, real=True)
    fid.update(fake_B_images_tensor, real=False)
    print("FID Score: ", fid.compute())

    # ssim_rec_real = []
    # ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(model.device)
    # for i in range(0, len(real_A_images)):
    #     ssim_rec_real.append(ssim(real_A_images[i].cpu(), cycle_images[i].cpu()))

    # ssim_rec_real_cpu = [x.cpu().numpy() for x in ssim_rec_real]

    # # Plotting a basic histogram
    # plt.hist(ssim_rec_real_cpu, bins=15, color='skyblue', edgecolor='black')
    
    # # Adding labels and title
    # plt.xlabel('SSIM')
    # plt.ylabel('Frequency')
    # plt.title('SSIM for A vs. F(G(A)) Distribution')
    
    # # Display the plot
    # plt.show()