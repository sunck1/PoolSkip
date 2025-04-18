import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class pool(nn.Module):
    """
    2D Maxpooling and Unpooling
    """
    def __init__(self, size=2):
        super(pool, self).__init__()
        self.pool = nn.MaxPool2d(size, stride=size, return_indices=True)
        self.unpool = nn.MaxUnpool2d(size, stride=size)

    def forward(self, x):
        output, indices = self.pool(x)
        out = self.unpool(output, indices)
        return out

class pool_skip(nn.Module):
    """
    The main function for PoolSkip.
    """
    def __init__(self, input_channel, output_channel, stride=1, kernel_size=3, padding=1, pool_size=2):
        super(pool_skip, self).__init__()
        self.pool_func = pool(pool_size)
        self.conv = nn.Conv2d(input_channel, output_channel, kernel_size=kernel_size, stride=stride, padding=padding)

    def l2_l1_ratio(self, weight):
        # Used for visualization: computes the ratio of L2 norm to L1 norm of the weight tensor
        l1_norm = torch.norm(weight, p=1)
        l2_norm = torch.norm(weight, p=2)
        ratio = l2_norm / l1_norm     
        return ratio.item()
        
    def forward(self, x):
        x = self.conv(self.pool_func(x)) + x
        return x
