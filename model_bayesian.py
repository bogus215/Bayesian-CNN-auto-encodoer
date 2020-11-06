# -*- coding: utf-8 -*-
import collections
from torch.distributions.normal import Normal
from layers.base import Flatten
import torch
import torch.nn as nn
from layers.base import Reshape
from mc_dropout import MCDropout


class Decoder(nn.Module):
    def __init__(self,
                 in_channels: int,
                 output_shape: tuple,
                 latent_dim: int,
                 drop_out_rate: float,
                 num_blocks: int = 4,
                 **kwargs):  # pylint: disable=unused-argument
        super(Decoder, self).__init__()
        self.in_channels = in_channels
        self.output_shape = output_shape
        self.out_channels = output_shape[0]
        self.latent_dim = latent_dim
        self.num_blocks = num_blocks
        self.drop_out_rate = drop_out_rate
        self.layers = nn.Sequential(
            collections.OrderedDict(
                [
                    ('linear', nn.Linear(self.latent_dim, self.in_channels)),
                    ('mc_dropout' , MCDropout( self.drop_out_rate , True) ),
                    ('reshape', Reshape(new_shape=(self.in_channels, 1, 1))),
                    ('reverse_gap', nn.UpsamplingBilinear2d(scale_factor=self.conv_input_shape[1])),
                    ('conv', self.make_layers(self.in_channels, self.out_channels, self.drop_out_rate ,1, self.num_blocks)),
                ]
            )
        )

    def forward(self, x):
        return self.layers(x)  # [0,1]

    @classmethod
    def make_layers(cls,
                    in_channels: int,
                    out_channels: int,
                    drop_out_rate: float,
                    num_convs: int = 1,
                    num_blocks: int = 4):

        layers = []
        for _ in range(num_blocks - 1):
            blk_kwargs = dict(
                in_channels=in_channels,
                out_channels=in_channels // 2,
                n=num_convs,
                drop_out_rate = drop_out_rate,
                activation='lrelu',
            )
            layers += [cls.make_block(**blk_kwargs)]
            in_channels = in_channels // 2

        # Final layer
        final_blk_kwargs = dict(
            in_channels=in_channels,
            out_channels=out_channels,
            n=num_convs,
            drop_out_rate=0, # last dropout No.
            activation='relu',  # [0, inf)
        )
        layers += [cls.make_block(**final_blk_kwargs)]

        return nn.Sequential(*layers)

    @staticmethod
    def make_block(in_channels: int, out_channels: int, n: int,drop_out_rate: float , activation: str = 'relu'):
        """Convolutional decoder block, with upsampling."""

        if activation == 'relu':
            act_fn = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            act_fn = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'sigmoid':
            act_fn = nn.Sigmoid()
        elif activation == 'tanh':
            act_fn = nn.Tanh()
        else:
            raise ValueError

        block = nn.Sequential()
        strides = [2] + [1] * (n-1)
        for i, s in enumerate(strides):
            conv_kwargs = {'kernel_size': 4, 'padding': s-1, 'stride': s, 'bias': False}
            block.add_module(f'conv{i}', nn.ConvTranspose2d(in_channels, out_channels, **conv_kwargs))
            block.add_module(f'bnorm{i}', nn.BatchNorm2d(out_channels))
            block.add_module(f'act{i}', act_fn)
            block.add_module(f'mc_dropout{i}', MCDropout(drop_out_rate, True))
            in_channels = out_channels

        return block

    @property
    def conv_input_shape(self):
        return tuple([self.in_channels] + [s // (2**self.num_blocks) for s in self.output_shape[1:]])


class Encoder(nn.Module):
    """Convolutional encoder."""
    def __init__(self,
                 input_shape: int,
                 out_channels: int,
                 latent_dim: int,
                 drop_out_rate: float,
                 num_blocks: int = 4,
                 **kwargs):  # pylint: disable=unused-argument

        super(Encoder, self).__init__()
        self.input_shape = input_shape     # (C_in, H, W)
        self.in_channels = input_shape[0]  # C_in
        self.out_channels = out_channels   # C_out
        self.latent_dim = latent_dim       # D_l
        self.num_blocks = num_blocks
        self.drop_out_rate = drop_out_rate
        self.layers = nn.Sequential(
            collections.OrderedDict(
                [('conv', self.make_layers(self.in_channels, self.out_channels, self.drop_out_rate,1, self.num_blocks))
                    ,('gap', nn.AdaptiveAvgPool2d(1))
                    ,('flatten', Flatten())
                ]
            )
        )

    def forward(self, x):  # pylint: disable=arguments-differ
        return self.layers(x)

    @classmethod
    def make_layers(cls,
                    in_channels: int,
                    out_channels: int,
                    drop_out_rate: float,
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
                drop_out_rate=drop_out_rate,
                n=num_convs,
                activation='lrelu'
            )
            layers = [cls.make_block(**blk_kwargs)] + layers  # insert to the front (LIFO)
            out_channels = out_channels // 2

        # First layer
        first_blk_kwargs = dict(
            in_channels=in_channels,
            out_channels=out_channels,
            drop_out_rate=drop_out_rate,
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
    def make_block(in_channels: int, out_channels: int, drop_out_rate: float,n: int,activation: str = 'relu'):
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
            block.add_module(f'mc_dropout{i}', MCDropout(drop_out_rate ,True))
            in_channels = out_channels

        return block


class Bottleneck(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 drop_out_rate: float,
                 reparameterize: bool = True
                 ):

        super(Bottleneck, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.reparameterize = reparameterize
        self.mu = nn.Linear(self.in_features, self.out_features)
        self.drop_out_rate = drop_out_rate
        self.mc_dropout = MCDropout(self.drop_out_rate , True)

        if self.reparameterize:
            self.logvar = nn.Linear(self.in_features, self.out_features)

    def forward(self, x):

        if self.reparameterize:
            mu, logvar = self.mu(x), self.logvar(x)
            z = self.reparameterization(mu, logvar)
        else:
            z = self.mu(x)
            z = self.mc_dropout(z)
        return z

    @staticmethod
    def reparameterization(mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(logvar / 2)
        sampled_z = Normal(mu, std).rsample()
        return sampled_z


class Model(nn.Module):

    def __init__(self, args, **kwargs):  # pylint: disable=unused-argument

        super(Model, self).__init__()

        input_shape = (60 + 45, args.size, args.size)  # Te 60 , Pro 45
        self.E = Encoder(input_shape=input_shape,
                         out_channels=512,
                         latent_dim=args.latent_dim,  # i.e. 100
                         reparameterize=args.reparameterize,
                         drop_out_rate = args.drop_out_rate
                         )

        self.B = Bottleneck(in_features=self.E.out_channels,
                            out_features=args.latent_dim,
                            reparameterize=args.reparameterize,
                            drop_out_rate=args.drop_out_rate)

        self.D = Decoder(in_channels=512,
                         output_shape=input_shape,
                         latent_dim=args.latent_dim,
                         reparameterize=args.reparameterize,
                         drop_out_rate=args.drop_out_rate
                         )

        self.D_var = Decoder(in_channels=512,
                         output_shape=input_shape,
                         latent_dim=args.latent_dim,
                         reparameterize=args.reparameterize,
                         drop_out_rate=args.drop_out_rate
                         )

    def forward(self, x):
        x_1 = self.E.layers[0][0](x)
        x_2 = self.E.layers[0][1](x_1)
        x_3 = self.E.layers[0][2](x_2)
        x_4 = self.E.layers[0][3](x_3)
        x_5 = self.E.layers[1:](x_4)

        x_5 = self.B(x_5)

        y_5 = self.D.layers[:4](x_5)
        y_4 = self.D.layers[4][0](y_5 + x_4)
        y_3 = self.D.layers[4][1](y_4 + x_3)
        y_2 = self.D.layers[4][2](y_3 + x_2)
        y = self.D.layers[4][3](y_2 + x_1)

        y_5_v = self.D_var.layers[:4](x_5)
        y_4_v = self.D_var.layers[4][0](y_5_v + x_4)
        y_3_v = self.D_var.layers[4][1](y_4_v + x_3)
        y_2_v = self.D_var.layers[4][2](y_3_v + x_2)
        y_var = self.D_var.layers[4][3](y_2_v + x_1)

        return y , y_var

    def initialize_weights(self):

        for model in [self.E, self.D]:

            for _, m in model.named_modules():

                if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                    nn.init.normal_(m.weight, mean=.0, std=.02)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.)

                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 1)

                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.02)
                    nn.init.constant_(m.bias, 0)

class MCDropout(nn.Module):
    def __init__(self, p: float = 0.5, force_dropout: bool = False):
        super().__init__()
        self.force_dropout = force_dropout
        self.p = p

    def forward(self, x):
        return nn.functional.dropout(x, p=self.p, training=self.training or self.force_dropout)
