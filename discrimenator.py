import torch
from torch import nn
import torch.nn.functional as F


class Discrimenator(nn.Module):
    def __init__(self, in_img, out_img):
        super(Discrimenator, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_img, out_img, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(out_img, out_img*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_img*2),
            nn.LeakyReLU(0.2),

            nn.Conv2d(out_img*2, out_img*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_img*4),
            nn.LeakyReLU(0.2),

            nn.Conv2d(out_img*4, out_img*8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_img*8),
            nn.LeakyReLU(0.2),

            nn.Conv2d(out_img*8, 1, kernel_size=4, stride=2, padding=0),
            nn.Sigmoid()
        ) 
        
    def forward(self, x):
        return self.conv(x)

