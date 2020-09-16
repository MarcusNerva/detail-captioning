#!/usr/bin/env python
# coding=utf-8
from dataset import DatasetMSRVTT, collate_fn
from torch.utils.data.dataloader import DataLoader
import sys
sys.path.append('../')
from mycfgs import get_total_settings

if __name__ == '__main__':
    args = get_total_settings()
    batch_size = args.batch_size

    dataset = DatasetMSRVTT('valid', args)
    dataloader = DataLoader(dataset, batch_size, shuffle=False, collate_fn=collate_fn)

    for i, (res2d, i3d, relation, object_, res2d_mask, i3d_mask, numberic, mask, seq) in enumerate(dataloader):
        print(res2d.shape)
        print(i3d.shape)
        print(relation.shape)
        print(object_.shape)
        print(res2d_mask.shape)
        print(i3d_mask.shape)
        print(numberic.shape)
        print(seq)

        if i > 3: break


