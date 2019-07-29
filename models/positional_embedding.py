# -*- coding: utf-8 -*-

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_positions(tensor, pad_idx):
    mask = tensor.ne(pad_idx).long()
    return torch.cumsum(mask, dim=0) * mask + pad_idx


# class LearnedPositionalEmbedding(nn.Embedding):
#     def __init__(self, num_emb, emb_dim, padding_idx):
#         super().__init__(num_emb, emb_dim, padding_idx)
# 
#     def forward(self, input, incremental_state=None):
#         if incremental_state is not None:
#             positions = input.data.new(1, 1).fill_(self.padding_idx + input.size(1))
#         else:
#             positions = make_positions(input.data, self.padding_idx)
#         return super().forward(positions)


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, embed_dim, pad_idx, init_size=1024):
        super().__init__()
        self.embed_dim = embed_dim
        self.pad_idx = pad_idx
        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            init_size, embed_dim, pad_idx)

    def get_embedding(num_emb, embed_dim, pad_idx=None):
        half_dim = embed_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_emb, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_emb, -1)
        if embed_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(num_emb, 1)], dim=1)
        if pad_idx is not None:
            emb[pad_idx, :] = 0
        return emb

    def forward(self, input, incremental_state=None, timestep=None):
        slen, bsz = input.size()
        max_pos = self.pad_idx + 1 + slen
        if self.weights is None or max_pos > self.weights.size(0):
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos, self.embed_dim, self.pad_idx)
      
        if incremental_state is not None:
            pos = (timestep.int() + 1).long() if timestep is not None else slen
            return self.weights[self.pad_idx+pos, :].expand(1, bsz, -1).to(input.device).detach()

        positions = make_positions(input, self.pad_idx)
        positions = self.weights.index_select(0, positions.view(-1).cpu())
        return positions.view(slen, bsz, -1).to(input.device).detach()

