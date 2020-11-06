# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['VaswaniAttention']


class VaswaniAttention(nn.Module):
    def __init__(self, hidden_size, context_size):
        super(VaswaniAttention, self).__init__()
        self.hidden_size = hidden_size
        self.context_size = context_size
        self.query_op = nn.Linear(self.hidden_size, self.context_size)
        self.key_op = nn.Linear(self.hidden_size, self.context_size)

    def forward(self, encoder_outputs, last_hidden):
        """
        Arguments:
            encoder_outputs: 3d tensor of shape (B, T, hidden_size).
            last_hidden: 2d tensor of shape (B, hidden_size).
        """
        query = self.query_op(last_hidden)          # (B, context_size)
        query = query.unsqueeze(-1)                 # (B, context_size, 1)
        keys = self.key_op(encoder_outputs)         # (B, T, context_size)
        scores = torch.bmm(keys, query)             # (B, T, 1)
        scores /= math.sqrt(self.hidden_size)
        scores = F.softmax(scores, dim=1)

        return scores.squeeze(-1)
