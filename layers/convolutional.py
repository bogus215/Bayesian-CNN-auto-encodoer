# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['SimpleConv', 'BasicBlock3D', 'BottleNeck3D']


class BasicBlock3D(nn.Module):
    """Add class docstring."""
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock3D, self).__init__()

        self.in_ch = in_channels
        self.out_ch = out_channels
        self.downsample = downsample
        self.stride = stride

        self.conv1 = nn.Conv3d(self.in_ch, self.out_ch, 3, 1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(self.out_ch)
        self.conv2 = nn.Conv3d(self.out_ch, self.out_ch, 3, 1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(self.out_ch)

        self.conv_skip = nn.Conv3d(self.in_ch, self.out_ch, 1, 1, padding=0, bias=True)

    def forward(self, x):

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += self.conv_skip(residual)
        out = F.relu(out)

        return out


class BottleNeck3D(nn.Module):
    """Add class docstring."""
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BottleNeck3D, self).__init__()

        self.in_ch = in_channels
        self.out_ch = out_channels

        self.conv1 = nn.Conv3d(self.in_ch, self.out_ch, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(self.out_ch)
        self.conv2 = nn.Conv3d(self.out_ch, self.out_ch, 3, stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(self.out_ch)
        self.conv3 = nn.Conv3d(self.out_ch, self.out_ch * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(self.out_ch * 4)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SimpleConv(nn.Module):
    """Add class docstring."""
    def __init__(self, in_channels, output_size=100):
        super(SimpleConv, self).__init__()

        self.in_channels = in_channels
        self.output_size = output_size

        self.conv_layers = nn.ModuleDict()
        self.conv_layers['conv1'] = nn.Conv2d(in_channels, 32, 3, stride=1, padding=1)
        self.conv_layers['conv2'] = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.conv_layers['conv3'] = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv_layers['conv4'] = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv_layers['conv5'] = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.conv_layers['conv6'] = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.conv_layers['conv7'] = nn.Conv2d(256, 100, 1, stride=1, padding=0)

        self.pool_layers = nn.ModuleDict()
        self.pool_layers['pool0'] = nn.MaxPool2d(2)
        self.pool_layers['pool1'] = nn.MaxPool2d(2)
        self.pool_layers['pool2'] = nn.MaxPool2d(2)
        self.pool_layers['pool3'] = nn.AvgPool2d(2)

        self.linear = nn.Linear(100 * 8 * 8, self.output_size)

    def forward(self, x):
        """
        Arguments:
            x: 4d tensor of shape (B, C, H, W).
        Returns 2d tensor of shape (B, output_size).
        """

        # (B, ?, 128, 128) -> (B, ?, 64, 64)
        out = self.pool_layers['pool0'](x) 

        # (B, ?, 64, 64) -> (B, 64, 32, 32)
        out = self.conv_layers['conv1'](out)
        out = F.relu(out)
        out = self.conv_layers['conv2'](out)
        out = F.relu(out)
        out = self.pool_layers['pool1'](out)

        # (B, 64, 32, 32) -> (B, 128, 16, 16)
        out = self.conv_layers['conv3'](out)
        out = F.relu(out)
        out = self.conv_layers['conv4'](out)
        out = F.relu(out)
        out = self.pool_layers['pool2'](out)

        # (B, 128, 16, 16) -> (B, 256, 8, 8)
        out = self.conv_layers['conv5'](out)
        out = F.relu(out)
        out = self.conv_layers['conv6'](out)
        out = F.relu(out)
        out = self.pool_layers['pool3'](out)

        # (B, 256, 8, 8) -> (B, 100, 8, 8)
        out = self.conv_layers['conv7'](out)

        # (B, 100, 8, 8) -> (B, output_size)
        out = out.view(x.size(0), -1)
        out = self.linear(out)
        out = F.relu(out)

        return out
