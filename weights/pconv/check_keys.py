import torch
weights = torch.load('weights/pconv/unet/model_weights.pth')
print(weights.keys())