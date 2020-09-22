import os
import shutil
import glob
import numpy as np
import torch

import sys
sys.path.append('../../')
sys.path.append('../../../')
from mymodels import DPPModel
from mycfgs.cfgs import get_total_settings

if __name__ == '__main__':
    args = get_total_settings()
    i3d_dir = args.i3d_dir

    settings = vars(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mode = 'i3d'
    
    i3d_list = glob.glob(os.path.join(i3d_dir, '*.npy'))
    i3d_list = sorted(i3d_list, key=lambda i3d_path: int(i3d_path.split('/')[-1].split('.')[0][5:]))
    i3d_mask = np.zeros((len(i3d_list), args.length), dtype=bool)

    for i, (i3d_path) in enumerate(i3d_list):
        dpp = DPPModel(mode, i3d_path, settings, device)
        Yg = dpp.dpp()
        Yg = sorted(Yg)
        i3d_mask[i, :len(Yg)] = True

        old_i3d_feats = np.load(i3d_path)
        new_i3d_feats = np.ascontiguousarray(old_i3d_feats[np.array(Yg)])
        i3d_feats = np.zeros((args.length, args.i3d_size))
        i3d_feats[:new_i3d_feats.shape[0], ...] = new_i3d_feats
        np.save(i3d_path, i3d_feats)

    np.save(args.i3d_mask_path, i3d_mask)
