import torch
from torch import nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, channels_noise, channels_img, feature_g):
        super(Generator, self).__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(channels_noise, feature_g*16, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(feature_g*16),
            nn.ReLU(),
            
            nn.ConvTranspose2d(feature_g*16, feature_g*8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_g*8),
            nn.ReLU(),
            
            nn.ConvTranspose2d(feature_g*8, feature_g*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_g*4),
            nn.ReLU(),
            
            nn.ConvTranspose2d(feature_g*4, feature_g*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_g*2),
            nn.ReLU(),
            
            nn.ConvTranspose2d(feature_g*2, channels_img, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )
        
    def forward(self, x):
        return self.conv(x)
    
