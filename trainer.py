# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class Trainer(object):
    def __init__(self, task, optimizer, scheduler, clip, n_iter=0):
        self.task = task
        self.model = task.model
        self.criterion = task.criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.clip = clip
        self.n_updates = n_iter

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def step(self, srcs, tgts, refs):
        self.optimizer.zero_grad()
        loss = self.task.loss(srcs, tgts, refs)
        if self.model.training:
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            self.optimizer.step()
            self.n_updates += 1
        return loss


class Task(object):
    def __init__(self, model, criterion):
        self.model = model
        self.criterion = criterion

    def loss(self, srcs, tgts, refs):
        raise NotImplementedError


class TranslationLM(Task):
    def __init__(self, model, criterion):
        super().__init__(model, criterion)

    def loss(self, srcs, tgts, refs=None):
        slen, bsz = srcs.size()

        outs = self.model(srcs, tgts[:-1])
        loss = self.criterion(
            outs[slen:].view(-1, outs.size(2)), 
            tgts.view(-1)
        )
        return loss
