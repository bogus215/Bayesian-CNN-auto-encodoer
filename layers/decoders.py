# -*- coding: utf-8 -*-

"""..."""

import collections

import torch
import torch.nn as nn

from layers.base import Reshape


class Decoder(nn.Module):
    def __init__(self,
                 in_channels: int,
                 output_shape: tuple,
                 latent_dim: int,
                 num_blocks: int = 4,
                 **kwargs):  # pylint: disable=unused-argument
        super(Decoder, self).__init__()
        self.in_channels = in_channels
        self.output_shape = output_shape
        self.out_channels = output_shape[0]
        self.latent_dim = latent_dim
        self.num_blocks = num_blocks
        self.layers = nn.Sequential(
            collections.OrderedDict(
                [
                    ('linear', nn.Linear(self.latent_dim, self.in_channels)),
                    ('reshape', Reshape(new_shape=(self.in_channels, 1, 1))),
                    ('reverse_gap', nn.UpsamplingBilinear2d(scale_factor=self.conv_input_shape[1])),
                    ('conv', self.make_layers(self.in_channels, self.out_channels, 1, self.num_blocks)),
                    ('tanh', nn.Tanh())
                ]
            )
        )

    def forward(self, x):
        return self.layers(x)  # [0,1]

    @classmethod
    def make_layers(cls,
                    in_channels: int,
                    out_channels: int,
                    num_convs: int = 1,
                    num_blocks: int = 4):

        layers = []
        for _ in range(num_blocks - 1):
            blk_kwargs = dict(
                in_channels=in_channels,
                out_channels=in_channels // 2,
                n=num_convs,
                activation='lrelu',
            )
            layers += [cls.make_block(**blk_kwargs)]
            in_channels = in_channels // 2

        # Final layer
        final_blk_kwargs = dict(
            in_channels=in_channels,
            out_channels=out_channels,
            n=num_convs,
            activation='lrelu',
        )
        layers += [cls.make_block(**final_blk_kwargs)]

        return nn.Sequential(*layers)

    @staticmethod
    def make_block(in_channels: int, out_channels: int, n: int, activation: str = 'relu'):
        """Convolutional decoder block, with upsampling."""

        if activation == 'relu':
            act_fn = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            act_fn = nn.LeakyReLU(0.2, inplace=True)
        else:
            raise ValueError

        block = nn.Sequential()
        strides = [2] + [1] * (n-1)
        for i, s in enumerate(strides):
            conv_kwargs = {'kernel_size': 4, 'padding': s-1, 'stride': s, 'bias': False}
            block.add_module(f'conv{i}', nn.ConvTranspose2d(in_channels, out_channels, **conv_kwargs))
            block.add_module(f'bnorm{i}', nn.BatchNorm2d(out_channels))
            block.add_module(f'act{i}', act_fn)
            in_channels = out_channels

        return block

    @property
    def conv_input_shape(self):
        return tuple([self.in_channels] + [s // (2**self.num_blocks) for s in self.output_shape[1:]])
