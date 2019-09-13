# -*- coding: utf-8 -*-

import argparse
import math
import os
import dill

from collections import OrderedDict

from tqdm import tqdm

from torchtext import data
from torchtext import datasets
from torchtext.vocab import Vectors

import torch
import torch.nn as nn
import torch.optim as optim

import options
import utils

from models.transformer import TranslationLM


def step(epoch, model, iterator, criterion, optimizer,  device):
    pbar = tqdm(iterator, dynamic_ncols=True) if model.training else iterator
    total_loss = 0.0
    for samples in pbar:
        optimizer.zero_grad()
        srcs = samples.text.to(device)
        tgts = samples.target.to(device)
        dec_outs = model(srcs)
        loss = criterion(
            dec_outs.view(-1, dec_outs.size(2)), 
            tgts.view(-1)
        )
        total_loss += loss.item()

        if model.training:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

            # setting of progressbar
            pbar.set_description(f'epoch {str(epoch).zfill(3)}')
            progress_state = OrderedDict(
                loss=loss.item(),
                ppl=math.exp(loss.item()),
                bsz=srcs.size(1),
                lr=optimizer.param_groups[0]['lr'], 
                clip=args.clip)
            pbar.set_postfix(progress_state)
    
    if model.training:
        pbar.close()
    total_loss /= len(iterator)

    mode = 'train' if model.training else 'valid'
    print(f'| epoch {str(epoch).zfill(3)} | {mode} ', end='') 
    print(f'| loss {total_loss:.{4}} ', end='')
    print(f'| ppl {math.exp(total_loss):.{4}} |', end='')
    print('')
    return total_loss


def main(args):
    device = torch.device('cuda' if args.gpu  else 'cpu')

    if args.re_training is None:
        TEXT = data.Field(
            lower=True,
            init_token='<bos>',
            pad_token='<pad>',
            eos_token='<eos>'
        )
    else: 
        basedir, _ = os.path.split(args.re_training)
        path = os.path.join(basedir, 'src.field')
        TEXT = utils.load_field(path)

    fields = [('text', TEXT)]

    slen_filter = lambda x: args.src_minlen <= len(x.src) <= args.src_maxlen \
                         and args.tgt_minlen <= len(x.tgt) <= args.tgt_maxlen

    # load training data
    train_data = datasets.LanguageModelingDataset(
        path=args.train, 
        text_field=TEXT, 
        newline_eos=True
    )

    # set Vocabulary object
    TEXT.build_vocab(
        train_data, 
        min_freq=args.min_freq, 
        specials=['<sep>'], 
    )

    if not os.path.exists(args.savedir):
        os.mkdir(args.savedir)
    utils.save_field(args.savedir, [('text', TEXT)])
    utils.save_vocab(args.savedir, [('text', TEXT)])

    # set training iterator
    train_iter = data.BPTTIterator(
        train_data, 
        batch_size=args.batch_size, 
        bptt_len=args.bptt_len,
        train=True, 
        repeat=False, 
        shuffle=True,
    )

    print(f'| [text] Dictionary: {len(TEXT.vocab.itos)} types')
    print('')

    print(f'train: {args.train}')
    utils.get_stats(train_iter, fields)

    # load validation data
    if args.valid is not None:
        valid_data = datasets.LanguageModelingDataset(
            path=args.valid,
            text_field=TEXT,
            newline_eos=True
        )

        valid_iter = data.BPTTIterator(
            valid_data, 
            batch_size=args.batch_size, 
            bptt_len=args.bptt_len,
            train=False,
            repeat=False, 
            shuffle=False,
        )

        print(f'valid: {args.valid}')
        utils.get_stats(valid_iter, fields)

    # build a model
    if  args.re_training is None:
        epoch = 1
        best_loss = math.inf
        model = TranslationLM(TEXT, args).to(device)
    else:
        load_vars = torch.load(args.re_training)
        epoch = load_vars['epoch'] + 1
        best_loss = load_vars['best_loss']
        lm_args, lm_weights = load_vars['args'], load_vars['weights']
        model = TranslationLM(TEXT, lm_args)
        model.load_state_dict(lm_weights)
        model.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=TEXT.vocab.stoi['<pad>'])

    optimizer_fn = utils.get_optimizer(args.optimizer)
    optimizer = optimizer_fn(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)

    # show the details of model and optimizer
    print('=============== MODEL ===============')
    print(model)
    print('')
    print('=============== OPTIMIZER ===============')
    print(optimizer)
    print('')
  
    max_epoch = (args.max_epoch or math.inf) + epoch

    while epoch < max_epoch and args.min_lr < optimizer.param_groups[0]['lr']:
        # training
        model.train()
        loss = step(epoch, model, train_iter, criterion, optimizer, device)

        # validation
        if args.valid is not None:
            model.eval()
            loss = step(epoch, model, valid_iter, criterion, optimizer, device)

        # saving model
        save_vars = {
            'epoch': epoch,
            'best_loss': loss if loss < best_loss else best_loss,
            'args': args, 
            'weights': model.state_dict()
        }

        if loss < best_loss:
            best_loss = loss
            filename = os.path.join(args.savedir, 'checkpoint_best.pt') 
            torch.save(save_vars, filename)
        if epoch % args.save_epoch == 0:
            filename = os.path.join(args.savedir, f'checkpoint_{epoch}.pt') 
            torch.save(save_vars, filename)
        filename = os.path.join(args.savedir, 'checkpoint_last.pt') 
        torch.save(save_vars, filename)

        # update
        scheduler.step(best_loss)
        epoch += 1

 
if __name__ == '__main__':
    parser = argparse.ArgumentParser('''
    ''')

    options.pretrain_opts(parser)
    options.model_opts(parser)
    args = parser.parse_args()
    main(args)
