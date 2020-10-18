#!/usr/bin/env python
# coding=utf-8
import numpy as np
import os
import glob
import sys
from collections import defaultdict
sys.path.append('../../')
sys.path.append('../../../')
from mycfgs import get_total_settings
import gc

if __name__ == '__main__':
    args = get_total_settings()
    raw_res2d_dir = args.raw_res2d_dir
    res2d_dir = args.res2d_dir
    res2d_size = args.res2d_size

    if not os.path.exists(res2d_dir):
        os.makedirs(res2d_dir)

    feats_list = glob.glob(os.path.join(raw_res2d_dir, '*.npy'))
    feats_list = sorted(feats_list)
    feats_dict = defaultdict(list)

    for feat_path in feats_list:
        video_id = feat_path.split('/')[-1].split('.')[0].split('_')[0]
        video_id = int(video_id)
        single_feat = np.load(feat_path)
        feats_dict[video_id].append(single_feat)

    for i in range(10000):
        save_path = os.path.join(res2d_dir, 'video' + str(i) + '.npy')
        res2d_feat = np.zeros((20, res2d_size))
        temp_feat = np.stack(feats_dict[i], axis=0)

        res2d_feat[:temp_feat.shape[0]] = temp_feat
        np.save(save_path, res2d_feat)
        del res2d_feat
        gc.collect()