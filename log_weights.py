import csv
import torch

class WeightLogger:
    def __init__(self, filename_A, filename_B):
        self.filename_A = filename_A
        self.filename_B = filename_B
        
        # Open the files and initialize writers
        with open(self.filename_A, mode='a', newline='') as file:
            self.writer_A = csv.writer(file)
            print("Logger initialized for ", self.filename_A)
        
        with open(self.filename_B, mode='a', newline='') as file:
            self.writer_B = csv.writer(file)
            print("Logger initialized for ", self.filename_B)         

    def log_weights(self, epoch, model):
        row_A = [epoch]
        row_B = [epoch]

        # Check if it's the first epoch to write the headers
        if epoch == 1:
            header_A = ['Epoch']
            header_B = ['Epoch']

            for name, param in model.netG_A.module.model.named_parameters():
                if 'weight' in name:
                    header_A.append(f"{name}_mean")

            for name, param in model.netG_B.module.model.named_parameters():
                if 'weight' in name:
                    header_B.append(f"{name}_mean")

            # Write the headers
            with open(self.filename_A, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(header_A)
            with open(self.filename_B, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(header_B)

        # Calculate the mean of the square root of the squares of the weights of model.netG_A
        for name, param in model.netG_A.module.model.named_parameters():
            if 'weight' in name:
                row_A.append(torch.sqrt(param.data ** 2).mean().item())

        # Calculate the mean of the square root of the squares of the weights of model.netG_B
        for name, param in model.netG_B.module.model.named_parameters():
            if 'weight' in name:
                row_B.append(torch.sqrt(param.data ** 2).mean().item())

        # Log the values for model.netG_A
        with open(self.filename_A, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(row_A)

        # Log the values for model.netG_B
        with open(self.filename_B, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(row_B)
