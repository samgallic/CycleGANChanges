import torch

class WeightLogger:
    def __init__(self):
        self.weights_A = {}        
        self.weights_B = {}   

    def log_weights(self, model):
        # Calculate the mean of the square root of the squares of the weights of model.netG_A
        for name, param in model.netG_A.module.model.named_parameters():
            if 'weight' in name:
                self.weights_A["model_A/" + name + "_mean"] = torch.sqrt(param.data ** 2).mean().item()

        # Calculate the mean of the square root of the squares of the weights of model.netG_B
        for name, param in model.netG_B.module.model.named_parameters():
            if 'weight' in name:
                self.weights_B["model_B/" + name + "_mean"] = torch.sqrt(param.data ** 2).mean().item()

        return self.weights_A, self.weights_B
