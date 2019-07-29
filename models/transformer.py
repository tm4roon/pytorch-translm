# -*- coding: utf-8 -*-

import random
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from .decoder import TransformerDecoderforLM


class TransformerLM(nn.Module):
    def __init__(self, field, args):
        super(TransformerLM, self).__init__()
        self.field = field
        self.vocabsize = len(field.vocab.itos)
        self.pad_idx = field.vocab.stoi['<pad>']
        self.bos_idx = field.vocab.stoi['<bos>']
        self.eos_idx = field.vocab.stoi['<eos>']
        self.sep_idx = field.vocab.stoi['<sep>']

        self.decoder = TransformerDecoderforLM(field, args)

    def forward(self, srcs, tgts=None):
        dec_outs = self.decoder(srcs, tgts)
        return dec_outs

    def generate(self, srcs, maxlen):
        slen, bsz = srcs.size()
        
        prev_tokens = torch.cat(
            (srcs,
             torch.ones_like(srcs[0]).unsqueeze(0) * self.sep_idx,
             torch.ones_like(srcs[0]).unsqueeze(0) * self.bos_idx
            ))

        while len(prev_tokens)-slen < maxlen:
            output_tokens = self.decoder(
                prev_tokens, incremental_state=True)
            output_tokens = output_tokens.max(2)[1][-1].unsqueeze(0)
            prev_tokens = torch.cat(
                (prev_tokens, output_tokens), 0)
        return prev_tokens
