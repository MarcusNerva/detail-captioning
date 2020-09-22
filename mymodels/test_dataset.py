#!/usr/bin/env python
# coding=utf-8
import torch
from torch.utils.data.dataloader import DataLoader
import sys
sys.append('../')
from dataset import DatasetMSRVTT, collate_fn
from mycfgs.cfgs import get_total_settings

if __name__ == '__main__':
    args = get_total_settings()
    valid_dataset = DatasetMSRVTT('valid', args)
    valid_dataloader = DataLoader(valid_dataset, args.batch_size, shuffle=True, collate_fn=collate_fn)
    
    for i, (res2d, i3d, relation, object_, res2d_mask, i3d_mask, numberic, mask, seq) in enumerate(valid_dataloader):
        print(res2d.shape)
        print(i3d.shape)
        print(relation.shape)
        print(object_.shape)
        print(res2d_mask.shape)
        print(i3d_mask.shape)
        print(numberic.shape)
        print(mask.shape)

        if i > 3: break
