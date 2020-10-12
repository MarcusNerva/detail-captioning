import os
import glob
import numpy as np
import pickle
import torch
import sys
sys.path.append('../../')
sys.path.append('../../../')
from mymodels import DPPModel
from mycfgs import get_total_settings

if __name__ == '__main__':
    args = get_total_settings()
    msvd_data_dir = args.msvd_data_dir
    i3d_subdir = args.i3d_subdir
    vid_dict_subpath = args.vid_dict_subpath
    i3d_dir = os.path.join(msvd_data_dir, i3d_subdir)
    vid_dict_path = os.path.join(msvd_data_dir, vid_dict_subpath)

    settings = vars(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mode = 'i3d'

    with open(vid_dict_path, 'rb') as f:
        vid_dict = pickle.load(f)
    i3d_list = glob.glob(os.path.join(i3d_dir, '*.npy'))
    i3d_mask = np.zeros((len(i3d_list), args.length))

    for i, (i3d_path) in enumerate(i3d_list):
        video_name = i3d_path.split('/')[-1].split('.')[0]
        video_id = vid_dict[video_name]

        dpp = DPPModel(mode, i3d_path, settings, device)
        Yg = dpp.dpp()
        Yg = sorted(Yg)
        i3d_mask[video_id, :len(Yg)] = True

        old_i3d_feats = np.load(i3d_path)
        new_i3d_feats = np.ascontiguousarray(old_i3d_feats[np.array(Yg)])
        i3d_feats = np.zeros((args.length, args.i3d_size))
        i3d_feats[:new_i3d_feats.shape[0], ...] = new_i3d_feats
        np.save(i3d_path, i3d_feats)