import torch
import numpy as np

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            numpy.ndarray: Numpy array of the unnormalized image.
        """
        # Clone the tensor to avoid modifying the original
        temp = tensor.clone()
        
        # Unnormalize
        for t, m, s in zip(temp, self.mean, self.std):
            t.mul_(s).add_(m)
            
        # Convert to uint8 numpy array
        temp = torch.clamp(temp * 255, min=0, max=255)
        temp = temp.to(torch.uint8)
        
        # Rearrange dimensions and convert to numpy
        if len(temp.shape) == 4:  # batch of images
            temp = temp.permute(0, 2, 3, 1)
        else:  # single image
            temp = temp.permute(1, 2, 0)
            
        return temp.cpu().numpy()

unorm_batch = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))