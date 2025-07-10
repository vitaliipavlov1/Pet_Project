import os
import matplotlib.pyplot as plt

from tqdm import tqdm
import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms.v2 as tfs_v2
import torch.nn as nn
import torch.optim as optim

from PIL import Image

from glob import glob
import py7zr
import cv2




class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_dropout=False):
        super().__init__()
        if down:
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True)
            )
        else:
            self.block = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        self.use_dropout = use_dropout
        if use_dropout:
            self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.block(x)
        return self.dropout(x) if self.use_dropout else x




class UNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=64):
        super().__init__()

        # Encoder (downsampling path)
        self.down_block1 = UNetBlock(in_channels, features, down=True, use_dropout=False)   # 256 -> 128
        self.down_block2 = UNetBlock(features, features * 2, down=True)                      # 128 -> 64
        self.down_block3 = UNetBlock(features * 2, features * 4, down=True)                  # 64 -> 32
        self.down_block4 = UNetBlock(features * 4, features * 8, down=True)                  # 32 -> 16
        self.down_block5 = UNetBlock(features * 8, features * 8, down=True)                  # 16 -> 8
        self.down_block6 = UNetBlock(features * 8, features * 8, down=True)                  # 8 -> 4

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, 4, 2, 1),
            nn.ReLU(inplace=True)
        )  # 4 -> 2

        # Decoder (upsampling path)
        self.up_block1 = UNetBlock(features * 8, features * 8, down=False, use_dropout=True) # 2 -> 4
        self.up_block2 = UNetBlock(features * 8 * 2, features * 8, down=False, use_dropout=True) # 4 -> 8
        self.up_block3 = UNetBlock(features * 8 * 2, features * 8, down=False) # 8 -> 16
        self.up_block4 = UNetBlock(features * 8 * 2, features * 4, down=False) # 16 -> 32
        self.up_block5 = UNetBlock(features * 4 * 2, features * 2, down=False) # 32 -> 64
        self.up_block6 = UNetBlock(features * 2 * 2, features, down=False)     # 64 -> 128

        # Output layer
        self.final_layer_up = nn.Sequential(
            nn.ConvTranspose2d(features * 2, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )  # 128 -> 256

    def forward(self, x):
        # Encoder
        d1 = self.down_block1(x)
        d2 = self.down_block2(d1)
        d3 = self.down_block3(d2)
        d4 = self.down_block4(d3)
        d5 = self.down_block5(d4)
        d6 = self.down_block6(d5)

        # Bottleneck
        bottleneck = self.bottleneck(d6)

        # Decoder with skip connections
        up1 = self.up_block1(bottleneck)
        up2 = self.up_block2(torch.cat([up1, d6], dim=1))
        up3 = self.up_block3(torch.cat([up2, d5], dim=1))
        up4 = self.up_block4(torch.cat([up3, d4], dim=1))
        up5 = self.up_block5(torch.cat([up4, d3], dim=1))
        up6 = self.up_block6(torch.cat([up5, d2], dim=1))
        final = self.final_layer_up(torch.cat([up6, d1], dim=1))

        return final
