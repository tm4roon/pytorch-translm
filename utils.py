# -*- coding: utf-8 -*-

import dill
import os
from collections import OrderedDict

import torch
import torch.optim as optim

from torchtext.vocab import Vectors


def save_field(savedir, fields):
    for field in fields:
        name, obj = field
        save_path = os.path.join(savedir, f"{name}.field")
        with open(save_path, 'wb') as fout:
            dill.dump(obj, fout)


def save_vocab(savedir, fields):
    for field in fields:
        name, obj = field
        save_path = os.path.join(savedir, f"{name}_vocab.txt")
        with open(save_path, 'w') as fout:
            for w in obj.vocab.itos:
                fout.write(w + '\n')


def get_examples(samples, name):
    if name == 'text':
        return samples.text
    if name == 'src': 
        return samples.src
    elif name == 'tgt': 
        return samples.tgt




def get_stats(iterator, fields):
    def calc_stats(iterator, name, field):
        pad_idx = field.vocab.stoi['<pad>']
        unk_idx = field.vocab.stoi['<unk>']
        n_tokens = 0
        n_unk = 0 
        for samples in iterator:
            examples = get_examples(samples, name)
            n_tokens += torch.sum(examples.ne(pad_idx).view(-1)).item()
            n_unk += torch.sum(examples.eq(unk_idx).view(-1)).item()
        return n_tokens, n_unk

    for name, field in fields:
        n_tokens, n_unk = calc_stats(iterator, name, field)
        print(f'| [{name}] {n_tokens} tokens,', end='')
        print(f' coverage: {100*(n_tokens-n_unk)/n_tokens:.{4}}%')
    print('')


def load_vector(embed_path):
    basedir, filename = os.path.split(embed_path)
    vectors = Vectors(name=filename, cache=basedir)
    return vectors


def id2w(pred, field):
    sentence = [field.vocab.itos[i] for i in pred]
    if '<sep>' in sentence:
        sentence = sentence[sentence.index('<sep>')+2:]
    if '<eos>' in sentence:
        return ' '.join(sentence[:sentence.index('<eos>')])
    return ' '.join(sentence)


def load_field(path):
    with open(path, 'rb') as f:
        return dill.load(f)


def get_optimizer(method):
    if method == 'sgd':
        return optim.SGD
    elif method == 'adam':
        return optim.Adam
    elif method == 'adagrad':
        return optim.Adagrad


