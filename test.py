#!/usr/bin/env python
# coding=utf-8
import torch
import os
from mymodels import CaptionModel, DatasetMSRVTT, collate_fn
from mycfgs.cfgs import get_total_settings
from myscripts.eval import eval


if __name__ == '__name__':
    args = get_total_settings()
    checkpoints_path = os.path.join(args.checkpoints_dir, 'best_model.pth')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_dataset = DatasetMSRVTT('test', args)
    args.pad_idx = test_dataset.get_pad_idx()
    args.bos_idx = test_dataset.get_bos_idx()
    args.eos_idx = test_dataset.get_eos_idx()
    args.n_vocab = test_dataset.get_n_vocab()
    
    model = CaptionModel(args)
    model = model.load_state_dict(torch.load(checkpoints_path))
    model.to(device)

    language_state = eval(args, model, test_dataset, device, collate_fn)
    print(language_state)
