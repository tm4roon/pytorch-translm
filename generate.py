# -*- coding: utf-8 -*-

import argparse
import os
import dill
from tqdm import tqdm

import torch
from torchtext import data

from options import generate_opts
import utils
from models.transformer import TransformerLM


def main(args):
    device = torch.device('cuda' if args.gpu  else 'cpu')

    load_vars = torch.load(args.model)
    lm_args = load_vars['args']
    weights = load_vars['weights']

    dirname = os.path.dirname(args.model)
    TEXT = utils.load_field(os.path.join(dirname, 'text.field'))
    fields = [('src', TEXT), ('tgt', TEXT)]

    with open(args.input, 'r') as f:
        examples = [data.Example.fromlist([line], [('src', TEXT)]) for line in f]
    
    test_data = data.Dataset(examples, [('src', TEXT)])
    test_iter = data.Iterator(
        test_data,
        batch_size=args.batch_size,
        train=False, 
        shuffle=False,
        sort=False,
    )

    model = TransformerLM(TEXT, lm_args).to(device)
    model.load_state_dict(weights)
    
    model.eval()
    for samples in tqdm(test_iter, total=len(test_iter)):
        srcs = samples.src.to(device)
        outs = model.generate(srcs, args.maxlen).transpose(0, 1)
        sents = [utils.id2w(out, TEXT) for out in outs]
        print('\n'.join(sents))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    generate_opts(parser)
    args = parser.parse_args()
    main(args)
