# -*- coding: utf-8 -*-

import collections
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from layers.base import Flatten


class Encoder(nn.Module):
    """Convolutional encoder."""
    def __init__(self,
                 input_shape: int,
                 out_channels: int,
                 latent_dim: int,
                 num_blocks: int = 4,
                 **kwargs):  # pylint: disable=unused-argument

        super(Encoder, self).__init__()
        self.input_shape = input_shape     # (C_in, H, W)
        self.in_channels = input_shape[0]  # C_in
        self.out_channels = out_channels   # C_out
        self.latent_dim = latent_dim       # D_l
        self.num_blocks = num_blocks
        self.reparameterize = kwargs.get('reparameterize', False)
        self.layers = nn.Sequential(
            collections.OrderedDict(
                [
                    ('conv', self.make_layers(self.in_channels, self.out_channels, 1, self.num_blocks)),
                    ('gap', nn.AdaptiveAvgPool2d(1)),
                    ('flatten', Flatten()),
                    ('bottleneck', Bottleneck(self.out_channels, self.latent_dim, reparameterize=self.reparameterize))
                ]
            )
        )

    def forward(self, x):  # pylint: disable=arguments-differ
        return self.layers(x)

    @classmethod
    def make_layers(cls,
                    in_channels: int,
                    out_channels: int,
                    num_convs: int = 1,
                    num_blocks: int = 4):
        """
        Arguments:
            in_channels: int,
            out_channels: int,
            num_convs: int, number of convolutional layers within each block.
            num_blocks: int, number of convolutional blocks
        """
        layers = []

        # Second to last layers
        for _ in range(num_blocks - 1):
            blk_kwargs = dict(
                in_channels=out_channels // 2,
                out_channels=out_channels,
                n=num_convs,
                activation='lrelu'
            )
            layers = [cls.make_block(**blk_kwargs)] + layers  # insert to the front (LIFO)
            out_channels = out_channels // 2

        # First layer
        first_blk_kwargs = dict(
            in_channels=in_channels,
            out_channels=out_channels,
            n=num_convs,
            activation='lrelu'
        )
        layers = [cls.make_block(**first_blk_kwargs)] + layers

        return nn.Sequential(*layers)

    @property
    def conv_output_shape(self):
        return tuple([self.out_channels] + [s // (2**self.num_blocks) for s in self.input_shape[1:]])

    @property
    def output_shape(self):
        return (self.latent_dim, )

    @staticmethod
    def make_block(in_channels: int, out_channels: int, n: int, activation: str = 'relu'):
        """Convolutional block."""
        if activation == 'relu':
            act_fn = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            act_fn = nn.LeakyReLU(0.2, inplace=True)
        else:
            raise ValueError

        conv_kwargs = {'kernel_size': 3, 'padding': 1, 'bias': False}

        block = nn.Sequential()
        strides = [2] + [1] * (n-1)
        for i, s in enumerate(strides):
            block.add_module(f'conv{i}', nn.Conv2d(in_channels, out_channels, stride=s, **conv_kwargs))
            block.add_module(f'bnorm{i}', nn.BatchNorm2d(out_channels))
            block.add_module(f'act{i}', act_fn)
            in_channels = out_channels

        return block



class Bottleneck(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 reparameterize: bool = True):

        super(Bottleneck, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.reparameterize = reparameterize

        self.mu = nn.Linear(self.in_features, self.out_features)
        if self.reparameterize:
            self.logvar = nn.Linear(self.in_features, self.out_features)

    def forward(self, x):
        if self.reparameterize:
            mu, logvar = self.mu(x), self.logvar(x)
            z = self.reparameterization(mu, logvar)
        else:
            z = self.mu(x)
        return z

    @staticmethod
    def reparameterization(mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(logvar/ 2)
        sampled_z = Normal(mu, std).rsample()  # TODO: sanity check
        return sampled_z
