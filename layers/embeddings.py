# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['ScalarEmbedding', 'CategoricalEmbedding']


class ScalarEmbedding(nn.Module):
    """Embedding layer for a scalar type spatial feature from pysc2."""
    def __init__(self, embedding_dim, name=None):
        super(ScalarEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.embed_fn = nn.Conv2d(
            in_channels=1,
            out_channels=self.embedding_dim,
            kernel_size=1,
            stride=1,
            padding=0
        )

        if name is not None:
            self.name = name
        else:
            self.name = self.__class__.__name__

    def forward(self, x):
        """
        Arguments:
            x: 4d tensor of shape (B, T, H, W)
        Returns a 5d tensor of shape (B, T, embedding_dim, H, W).
        """
        inputs = x.permute(1, 0, 2, 3)  # (T, B, H, W)
        outputs = []
        for inp in inputs:
            inp = inp.unsqueeze(1)    # (B, H, W) -> (B, 1, H, W)
            out = self.embed_fn(inp.float())  # (B, embedding_dim, H, W)
            outputs.append(out)

        outputs = torch.stack(outputs, dim=1)  # (B, T, embedding_dim, H, W)

        return outputs


class CategoricalEmbedding(nn.Module):
    """Embedding layer for a categorical spatial feature from pysc2."""
    def __init__(self, category_size, embedding_dim, name=None):
        super(CategoricalEmbedding, self).__init__()
        self.category_size = category_size
        self.embedding_dim = embedding_dim
        self.embed_fn = nn.Embedding(
            num_embeddings=category_size,
            embedding_dim=embedding_dim,
            padding_idx=0
        )

        if name is not None:
            self.name = name
        else:
            self.name = self.__class__.__name__

    def forward(self, x):
        """
        Arguments:
            x: 4d tensor of shape (B, T, H, W)
        Returns a 5d tensor of shape (B, T, embedding_dim, H, W).
        """
        try:
            out = self.embed_fn(x.long())
        except RuntimeError as e:
            print(f"Name: {self.name}")
            print(f"MAX: {x.max()}, MIN: {x.min()}")
            raise RuntimeError(str(e))
        out = out.permute(0, 1, 4, 2, 3)

        return out
