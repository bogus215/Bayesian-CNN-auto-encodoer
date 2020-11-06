# -*- coding: utf-8 -*-

"""Basic usage layers."""

import torch
import torch.nn as nn


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()  # pylint: disable=useless-super-delegation

    def forward(self, x):
        return x.view(x.size(0), -1)


class Reshape(nn.Module):
    def __init__(self, new_shape: tuple):
        super(Reshape, self).__init__()  # pylint: disable=useless-super-delegation
        if len(new_shape) != 3:
            raise ValueError
        self.new_shape = new_shape

    def forward(self, x: torch.Tensor):
        if x.ndim != 2:
            raise ValueError(f"Expects 2D input, but received {x.ndim}D.")
        return x.view(x.size(0), *self.new_shape)
