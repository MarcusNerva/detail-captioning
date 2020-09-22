#!/usr/bin/env python
# coding=utf-8
import numpy as np
import torch
import torchtext
import os
import pickle
import gc

def make_datastore(args, mode, data_range):
    if not mode in ['train', 'valid', 'test']:
        raise NotImplementedError

    res2d_dir = args.res2d_dir
    i3d_dir = args.i3d_dir
    relation_dir = args.relation_dir
    object_dir = args.object_dir
    seq_mask_path = args.seq_mask_path
    numberic_dict_path = args.numberic_dict_path

    if mode == 'train':
        output_dir = args.train_datastore_dir
    elif mode == 'valid':
        output_dir = args.valid_datastore_dir
    else:
        output_dir = args.test_datastore_dir
    
    res2d_path = os.path.join(output_dir, 'res2d.pkl')
    i3d_path = os.path.join(output_dir, 'i3d.pkl')
    relation_path = os.path.join(output_dir, 'relation.pkl')
    object_path = os.path.join(output_dir, 'object.pkl')
    numberic_path = os.path.join(output_dir, 'numberic.pkl')
    video_id_path = os.path.join(output_dir, 'video_id.pkl')
    word_mask_path = os.path.join(output_dir, 'word_mask.pkl')

    res2d_dict, i3d_dict, relation_dict, object_dict = {}, {}, {}, {}
    numberic, video_id, word_mask = [], [], []

    with open(numberic_dict_path, 'rb') as f:
        numberic_dict = pickle.load(f)
    with open(seq_mask_path, 'rb') as f:
        seq_mask = pickle.load(f)

    for vid in data_range:
        res2d_single_path = os.path.join(res2d_dir, vid + '.npy')
        res2d = np.load(res2d_single_path)
        res2d_dict[vid] = res2d
    with open(res2d_path, 'wb') as f:
        pickle.dump(res2d_dict, f)
    del res2d_dict
    gc.collect()

    for vid in data_range:
        i3d_single_path = os.path.join(i3d_dir, vid + '.npy')
        i3d = np.load(i3d_single_path)
        i3d_dict[vid] = i3d
    with open(i3d_path, 'wb') as f:
        pickle.dump(i3d_path, f)
    del i3d_dict
    gc.collect()

    for vid in data_range:
        relation_single_path = os.path.join(relation_dir, vid + '.npy')
        relation = np.load(relation_single_path)
        relation_dict[vid] = relation
    with open(relation_path, 'wb') as f:
        pickle.dump(relation_dict, f)
    del relation_dict
    gc.collect()

    for vid in data_range:
        object_single_path = os.path.join(object_dir, vid + '.npy')
        object_ = np.load(object_single_path)
        object_dict[vid] = object_
    with open(object_path, 'wb') as f:
        pickle.dump(object_dict, f)
    del object_dict
    gc.collect()

    for vid in data_range:
        for number_tensor in numberic_dict[vid]:
            numberic.append(number_tensor)
            video_id.append(vid)
        for mask in seq_mask[vid]:
            word_mask.append(mask)

    with open(numberic_path, 'wb') as f:
        pickle.dump(numberic, f)
    with open(video_id_path, 'wb') as f:
        pickle.dump(video_id, f)
    with open(word_mask_path, 'wb') as f:
        pickle.dump(word_mask, f)

    
