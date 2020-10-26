import numpy as np
import os
import pickle
import glob
import sys
from collections import defaultdict
sys.path.append('../../')
sys.path.append('../../../')
from mycfgs import get_total_settings
import gc

if __name__ == '__main__':
    args = get_total_settings()
    msvd_data_dir = args.msvd_data_dir
    vid_dict_subpath = args.vid_dict_subpath
    raw_res2d_subdir = args.raw_res2d_subdir
    res2d_subdir = args.res2d_subdir
    res2d_size = args.res2d_size

    raw_res2d_dir = os.path.join(msvd_data_dir, raw_res2d_subdir)
    res2d_dir = os.path.join(msvd_data_dir, res2d_subdir)
    vid_dict_path = os.path.join(msvd_data_dir, vid_dict_subpath)

    if not os.path.exists(res2d_dir):
        os.makedirs(res2d_dir)
    with open(vid_dict_path, 'rb') as f:
        vid_dict = pickle.load(f)

    video_name_list = list(vid_dict.keys())
    feats_list = glob.glob(os.path.join(raw_res2d_dir, '*.npy'))
    feats_list = sorted(feats_list)
    feats_dict = defaultdict(list)

    for feat_path in feats_list:
        video_name = feat_path.split('/')[-1].split('.')[0]
        video_name = video_name[:-7]
        assert video_name in video_name_list, 'in merge_msvd_scene_feats.py video_name generation is wrong!'
        single_feat = np.load(feat_path)
        feats_dict[video_name].append(single_feat)

    for video_name in video_name_list:
        save_path = os.path.join(res2d_dir, video_name + '.npy')
        res2d_feat = np.zeros((args.length, res2d_size))
        temp_feats = np.stack(feats_dict[video_name], axis=0)

        res2d_feat[:temp_feats.shape[0]] = temp_feats
        np.save(save_path, res2d_feat)
        del res2d_feat
        gc.collect()