#!/usr/bin/env python
# coding=utf-8
import torch
import os
from mymodels import CaptionModel, CaptionModel_Part, DatasetMSRVTT, collate_fn, DatasetMSVD, msvd_collate_fn
from mycfgs.cfgs import get_total_settings
from myscripts.eval import eval


if __name__ == '__main__':
    args = get_total_settings()
    part_model = args.part_model
    test_situation = args.test_situation
    args.beam_size = 3
    best_model_path = os.path.join(args.checkpoints_dir, 'best_model.pth')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_dataset = DatasetMSRVTT('test', args) if not args.now_msvd else DatasetMSVD('test', args)
    _collate_fn = collate_fn if not args.now_msvd else msvd_collate_fn

    args.pad_idx = test_dataset.get_pad_idx()
    args.bos_idx = test_dataset.get_bos_idx()
    args.eos_idx = test_dataset.get_eos_idx()
    args.unk_idx = test_dataset.get_unk_idx()
    args.n_vocab = test_dataset.get_n_vocab()
    
    model = CaptionModel_Part(args) if part_model else CaptionModel(args)
    model = model.double()
    model.to(device)
    model.load_state_dict(torch.load(best_model_path))

    language_state = eval(args, model, test_dataset, device, _collate_fn, test_situation)
    print(language_state)
