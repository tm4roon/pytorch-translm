# -*- coding: utf-8 -*-

import argparse
import math
import os
import dill

from collections import OrderedDict

from tqdm import tqdm

from torchtext import data
from torchtext import datasets

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
        srcs = samples.src.to(device)
        tgts = samples.tgt.to(device)
        dec_outs = model(srcs, tgts[:-1])
        loss = criterion(
            dec_outs.view(-1, dec_outs.size(2)), 
            tgts[1:].view(-1)
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

    if args.finetune: 
        basedir, _ = os.path.split(args.finetune)
        path = os.path.join(basedir, 'text.field')
        TEXT = utils.load_field(path)
    else:
        TEXT = data.Field(
            lower=True, 
            init_token='<bos>', 
            eos_token='<eos>'
        )

    fields = [('src', TEXT), ('tgt', TEXT)]

    slen_filter = lambda x: args.src_minlen <= len(x.src) <= args.src_maxlen \
                         and args.tgt_minlen <= len(x.tgt) <= args.tgt_maxlen

    # load training data
    train_data = data.TabularDataset(
            path=args.train,
            format='tsv',
            fields=fields,
            filter_pred=slen_filter,
    )

    # set Vocabulary object
    if args.finetune is None:
        TEXT.build_vocab(
            train_data, 
            min_freq=args.min_freq, 
            specials=['<sep>', '<mask>'], 
        )

    if not os.path.exists(args.savedir):
        os.mkdir(args.savedir)
    utils.save_field(args.savedir, [('text', TEXT)])
    utils.save_vocab(args.savedir, [('text', TEXT)])

    # set training iterator
    train_iter = data.BucketIterator(
        train_data, 
        batch_size=args.batch_size,
        sort_within_batch=True,
        sort_key= lambda x: len(x.src),
        repeat=False,
    )

    print(f'| [text] Dictionary: {len(TEXT.vocab.itos)} types')
    print('')

    print(f' train: {args.train}')
    utils.get_stats(train_iter, fields)

    # load validation data
    valid_data = data.TabularDataset(
        path=args.valid,
        format='tsv',
        fields=fields,
        filter_pred=slen_filter,
    )

    valid_iter = data.BucketIterator(
        valid_data, 
        batch_size=args.batch_size,
        sort_within_batch=True,
        sort_key= lambda x: len(x.src),
        train=False,
        repeat=False,
        shuffle=False
    )

    print(f'valid: {args.valid}')
    utils.get_stats(valid_iter, fields)

    # build a model
    if args.finetune:
        load_vars = torch.load(args.finetune)
        epoch = load_vars['epoch'] + 1
        best_loss = load_vars['best_loss']
        lm_args, lm_weights = load_vars['args'], load_vars['weights']
        model = TranslationLM(TEXT, lm_args)
        model.load_state_dict(lm_weights)
        model.to(device)
    else:
        epoch = 1
        best_loss = math.inf
        model = TranslationLM(TEXT, args).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=TEXT.vocab.stoi['<pad>'])

    optimizer_fn = utils.get_optimizer(args.optimizer)
    optimizer = optimizer_fn(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')

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
        _ = step(epoch, model, train_iter, criterion, optimizer, device)

        # validation
        model.eval()
        valid_loss = step(epoch, model, valid_iter, criterion, optimizer, device)

        # saving model
        save_vars = {
            'epoch': epoch,
            'best_loss': valid_loss if valid_loss < best_loss else best_loss,
            'args': args, 
            'weights': model.state_dict()
        }

        if valid_loss < best_loss:
            best_loss = valid_loss
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

    options.finetune_opts(parser)
    options.model_opts(parser)
    args = parser.parse_args()
    main(args)
