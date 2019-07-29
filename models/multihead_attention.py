# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from  torch.nn import Parameter
import torch.nn.functional as F


class MultiheadAttn(nn.Module):
    def __init__(self, embed_dim, n_heads,kdim=None, vdim=None, dout=0.0, 
        bias=True, add_bias_kv=False):
        super(MultiheadAttn, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.n_heads = n_heads
        self.dropout = dout
        self.head_dim = embed_dim // n_heads
        assert self.head_dim * n_heads == self.embed_dim, \
               "embed_dim must be divisible by num_heads"
        self.scale = self.head_dim ** -0.5

        if self.qkv_same_dim:
            self.in_weight = Parameter(torch.Tensor(3*embed_dim, embed_dim))
        else:
            self.k_weight = Parameter(torch.Tensor(embed_dim, self.kdim))
            self.v_weight = Parameter(torch.Tensor(embed_dim, self.vdim))
            self.q_weight = Parameter(torch.Tensor(embed_dim, embed_dim))

        self.in_bias = Parameter(torch.Tensor(3*embed_dim)) if bias else None

        if add_bias_kv:
            self.k_bias = Parameter(torch.Tensor(1, 1, embed_dim))
            self.v_bias = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.k_bias = self.v_bias = None

        self.out = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        if self.qkv_same_dim:
            nn.init.xavier_uniform_(self.in_weight)
        else:
            nn.init_xavier_uniform_(self.k_weight)
            nn.init_xavier_uniform_(self.v_weight)
            nn.init_xavier_uniform_(self.q_weight)
        nn.init.xavier_uniform_(self.out.weight)

        if self.in_bias is not None:
            nn.init.constant_(self.in_bias, 0.)
            nn.init.constant_(self.out.bias, 0.)
        if self.k_bias is not None:
            nn.init.xavier_normal_(self.k_bias)
        if self.v_bias is not None:
            nn.init_xavier_normal_(self.v_bias)
 

    def forward(self, query, key, value, key_pad_mask=None, incremental_state=None, 
        need_weights=True, static_kv=False, attn_mask=None, dec=False):
        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        kv_same = key.data_ptr() == value.data_ptr()

        tlen, bsz, embed_dim = query.size()

        if qkv_same: 
            q, k, v = self.in_projection_qkv(query)
        elif kv_same:
            q = self.in_projection_q(query)
            if key is None:
                k = v = None
            else:
                k = self.in_projection_k(key)
                v = self.in_projection_v(key)
        else:
            q = self.in_projection_q(query)
            k = self.in_projection_k(key)
            v = self.in_projection_v(value)
        q *= self.scale

        if self.k_bias is not None:
            k = torch.cat([k, self.k_bias.repeat(1, bsz, 1)])
            v = torch.cat([v, self.v_bias.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_pad_mask is not None:
                key_pad_mask = torch.cat(
                    [key_pad_mask, key_pad_mask.new_zeros(key_pad_mask.size(0), 1)], dim=1)

        q = q.contiguous().view(tlen, bsz*self.n_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz*self.n_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz*self.n_heads, self.head_dim).transpose(0, 1)

        slen = k.size(1)
        attn_weights = torch.bmm(q, k.transpose(1, 2))

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            attn_weights += attn_mask

        if key_pad_mask is not None:
            attn_weights = attn_weights.view(bsz, self.n_heads, tlen, slen)
            attn_weights = attn_weights.float().masked_fill(
                key_pad_mask.transpose(1, 0).unsqueeze(1).unsqueeze(2),
                float('-inf')
            ).type_as(attn_weights)
            attn_weights = attn_weights.view(bsz*self.n_heads, tlen, slen)

        attn_weights = F.softmax(attn_weights, dim=-1).type_as(attn_weights)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        
        attn = torch.bmm(attn_weights, v)
        attn = attn.transpose(0, 1).contiguous().view(tlen, bsz, embed_dim)
        attn = self.out(attn)

        if need_weights:
            attn_weights = attn_weights.view(bsz, self.n_heads, tlen, slen)
            attn_weights = attn_weights.sum(dim=1) / self.n_heads
        else:
            attn_weights = None
        return attn, attn_weights

    def in_projection_qkv(self, query):
        return self._in_projection(query).chunk(3, dim=-1)

    def in_projection_q(self, query):
        if self.qkv_same_dim:
            return self._in_projection(query, end=self.embed_dim)
        else:
            bias = self.in_bias
            if bias is not None:
                bias = bias[:self.embed_dim]
            return F.linear(query, self.q_weight, bias)

    def in_projection_k(self, key):
        if self.qkv_same_dim:
            return self._in_projection(
                    key, start=self.embed_dim, end=2*self.embed_dim)
        else:
            weight = self.k_weight
            bias = self.in_bias
            if bias is not None:
                bias = bias[self.embed_dim:2*self.embed_dim]
            return F.linear(key, weight, bias)

    def in_projection_v(self, value):
        if self.qkv_same_dim:
            return self._in_projection(value, start=2*self.embed_dim)
        else:
            weight = self.v_weight
            bias = self.in_bias
            if bias is not None:
                bias = bias[2*self.embed_dim:]
            return F.linear(value, weight, bias)

    def _in_projection(self, input, start=0, end=None):
        weight = self.in_weight
        bias = self.in_bias
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)
