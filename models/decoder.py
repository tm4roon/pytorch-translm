# -*- coding: utf-8 -*-

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .positional_embedding import SinusoidalPositionalEmbedding

from .multihead_attention import MultiheadAttn
from .utils import (
    fill_ninf,
    Linear,
)


class TransformerDecoder(nn.Module):
    def __init__(self, field, args, no_enc_attn=False):
        super(TransformerDecoder, self).__init__()
        self.field = field
        self.vocabsize = len(self.field.vocab.itos)
        self.pad_idx = self.field.vocab.stoi['<pad>']
        self.dropout = args.dropout
        self.embed_dim = args.embed_dim
        self.n_layers = args.layers

        self.w_embed = nn.Embedding(self.vocabsize, self.embed_dim) \
            if field.vocab.vectors is None \
            else nn.Embedding.from_pretrained(field.vocab.vectors, freeze=True)
        self.p_embed = SinusoidalPositionalEmbedding(self.embed_dim, self.pad_idx)
        self.embed_scale = math.sqrt(self.embed_dim)

        self.layers = nn.ModuleList(
            [TransformerDecoderLayer(args, no_enc_attn) for _ in range(self.n_layers)])
        self.out_projection = Linear(self.embed_dim, self.vocabsize)

    def forward(self, prev_tokens, enc_outs, incremental_state=None):
        # embed positions
        positions = self.p_embed(prev_tokens, incremental_state=incremental_state)

        x = self.embed_scale * self.w_embed(prev_tokens)
        x += positions
        x = F.dropout(x, p=self.dropout, training=self.training)

        # padding mask
        decoder_pad_mask = prev_tokens.eq(self.pad_idx)
        if not decoder_pad_mask.any():
            decoder_pad_mask = None

        # decoder layers
        for layer in self.layers:
            x = layer(
                x, 
                enc_outs['enc_out'] if enc_outs is not None else None,
                enc_outs['enc_pad_mask'] if enc_outs is not None else None,
                self.buffered_future_mask(x) if incremental_state is None else None,
                decoder_pad_mask,
                incremental_state,
            )
        x = self.out_projection(x)
        return x

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if not hasattr(self, '_future_mask') or self._future_mask is None:
            self._future_mask = torch.triu(fill_ninf(tensor.new(dim, dim)), 1)
        if self._future_mask.size(0) < dim:
            self._future_mask = torch.triu(
                fill_ninf(self._future_mask.resize_(dim, dim)), 1)
        return self._future_mask[:dim, :dim]


class TransformerDecoderLayer(nn.Module):
    def __init__(self, args, no_encoder_attn=False):
        super(TransformerDecoderLayer, self).__init__()
        self.dropout = args.dropout
        self.attention_dropout = args.attention_dropout
        self.activation_dropout = args.activation_dropout

        self.embed_dim = args.embed_dim
        self.hidden_dim = args.hidden_dim
        self.n_heads = args.heads

        self.normalize_before = args.normalize_before
        self.self_attn = MultiheadAttn(
            self.embed_dim, self.n_heads, dout=self.attention_dropout)
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        if no_encoder_attn:
            self.enc_attn = None
            self.enc_attn_layer_norm = None
        else:
            self.enc_attn = MultiheadAttn(
                self.embed_dim, self.n_heads, dout=self.attention_dropout)
            self.enc_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        self.fc1 = Linear(self.embed_dim, self.hidden_dim)
        self.fc2 = Linear(self.hidden_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(self, x, enc_out, enc_pad_mask, attn_mask, dec_pad_mask, incremental_state=None):
        residual = x
        x = self.maybe_normalize(self.self_attn_layer_norm, x, before=True)
        x, _ = self.self_attn(
            query=x, 
            key=x, 
            value=x, 
            key_pad_mask=dec_pad_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=attn_mask,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_normalize(self.self_attn_layer_norm, x, after=True)

        if self.enc_attn is not None:
            residual = x
            x = self.maybe_normalize(self.enc_attn_layer_norm, x, before=True)
            x, attn = self.enc_attn(
                query=x,
                key=enc_out,
                value=enc_out,
                key_pad_mask=enc_pad_mask,
                incremental_state=incremental_state,
                dec=True
            )
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = residual + x
            x = self.maybe_normalize(self.enc_attn_layer_norm, x, after=True)

        residual = x
        x = self.maybe_normalize(self.final_layer_norm, x, before=True)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_normalize(self.final_layer_norm, x, after=True)
        return x

    def maybe_normalize(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x
