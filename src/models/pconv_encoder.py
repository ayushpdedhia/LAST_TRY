import torch
import torch.nn.functional as F
from torch import nn
from .partial_conv2d import PartialConv2d

class PConvEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bn=True):
        super(PConvEncoder, self).__init__()
        self.pconv = PartialConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=2, padding=(kernel_size-1)//2,multi_channel=True, return_mask=True)
        self.bn = bn
        if bn:  # Only create batchnorm if bn=True
            self.batchnorm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()
        
    def forward(self, img, mask_in):
        conv, mask = self.pconv(img, mask_in)
        if self.bn:  # Only apply batchnorm if it exists
            conv = self.batchnorm(conv)
        conv = self.activation(conv)
        return conv, mask