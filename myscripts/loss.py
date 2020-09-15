#!/usr/bin/env python
# coding=utf-8
import torch
import torch.nn as nn
import numpy as np

def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()

class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, pred, target, mask, eos_idx):

        pred = to_contiguous(pred).view(-1, pred.size(-1))
        target = torch.cat([target[:, 1:], target[:, 0].unsqueeze(1).fill_(eos_idx)], dim=1)
        target = to_contiguous(target).view(-1, 1)
        mask = to_contiguous(mask).view(-1, 1)

        output = -1. * pred.gather(1, target) * mask
        output = torch.sum(output) / torch.sum(mask)
        return output

class RewardCriterion(nn.Module):
    def __init__(self):
        super(RewardCriterion, self).__init__()

    def forward(self, pred, seq, reward):
        pred = to_contiguous(pred).view(-1)
        reward = to_contiguous(reward).view(-1)
        mask = (seq > 0).float()
        mask = to_contiguous(torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], dim=1)).view(-1)
        output = -pred * reward * mask
        output = torch.sum(output) / torch.sum(mask)
        return output
