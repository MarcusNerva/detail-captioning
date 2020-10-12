#!/usr/bin/env python
# coding=utf-8
import os
import glob
import pickle
import numpy as np
import itertools

# _x, _y, _s, _permutations, _mat_mapping is initialized in main function

def softmax(matrix):
    """
    Shapes:
        matrix: (n_obj, n_obj)
    """
    e_m = np.exp(matrix - matrix.max(axis=1, keepdims=True))
    ret = e_m / e_m.sum(axis=1, keepdims=True)
    return ret


def iou(prev, cur):
    """
    Shapes:
        prev: (n_obj, 4)
        cur: (n_obj, 4)
    """
    n_prev_obj, n_cur_obj = prev.shape[0], cur.shape[0]

    box_prev, box_cur = prev[_x], cur[_y]
    # box_union = np.maximum(box_prev[:, 2:], box_cur[:, 2:]) - np.minimum(box_prev[:, :2], box_cur[:, :2])
    # union = box_union[:, 0] * box_union[:, 1]
    
    height = np.maximum(np.minimum(box_prev[:, 2], box_cur[:, 2]) - np.maximum(box_prev[:, 0], box_cur[:, 0]), 0)
    width = np.maximum(np.minimum(box_prev[:, 3], box_cur[:, 3]) - np.maximum(box_prev[:, 1], box_cur[:, 1]), 0)
    intersection = height * width

    prev_area = box_prev[:, 2:] - box_prev[:, :2]
    prev_area = prev_area[:, 0] * prev_area[:, 1]
    cur_area = box_cur[:, 2:] - box_cur[:, :2]
    cur_area = cur_area[:, 0] * cur_area[:, 1]
    union = prev_area + cur_area - intersection

    iou = intersection / union
    iou = iou.reshape(n_prev_obj, n_cur_obj)
    iou = softmax(iou)    
    return iou
    

def cosine_similar(prev, cur):
    """
    Shapes:
        prev: (n_obj, 1024)
        cur: (n_obj, 1024)
    """
    cosine = np.dot(prev, cur.T)
    cosine = softmax(cosine)
    return cosine


def get_similar_mat(cosine, iou, lambda_value):
    """
    similar_value is calculate by Weighted addition between cosine value and iou value.
    """
    similar_matrix = lambda_value * cosine + (1.0 - lambda_value) * iou
    similar_matrix = softmax(similar_matrix)
    return similar_matrix


def get_match(similar_matrix):
    """
    get the optimal match pattern via Brute-force method.
    """
    def get_value(permut):
        ret = 0
        for i in range(5):
            ret += similar_matrix[i][permut[i]]
        return ret

    def sgn(x):
        eps = 1e-6
        if abs(x) < eps: return 0
        elif x > 0: return 1
        return -1

    ret = None
    for permut in _permutations:
        if ret is None or sgn(get_value(ret) - get_value(permut)) < 0:
            ret = permut

    return ret


def rearrange(args, vid):
    msdn_features = args.msdn_features
    relation_features = args.relation_features
    object_features = args.object_features
    box_boundings = args.box_boundings
    rearranged_box_boundings = args.rearranged_box_boundings
    lambda_value = args.lambda_value
    now_msvd = args.now_msvd

    if not os.path.exists(relation_features):
        os.makedirs(relation_features)
    if not os.path.exists(object_features):
        os.makedirs(object_features)

    video_id = '%06d' % vid
    feat_path = os.path.join(msdn_features, video_id + '.npy' if not now_msvd else vid + '.npy')
    relation_feat_path = os.path.join(relation_features, 'video' + str(vid) + '.npy' if not now_msvd else vid + '.npy')
    object_feat_path = os.path.join(object_features, 'video' + str(vid) + '.npy' if not now_msvd else vid + '.npy')
    box_path = os.path.join(box_boundings, 'video' + str(vid) + '.npy' if not now_msvd else vid + '.npy')
    # rearranged_box_path = os.path.join(rearranged_box_boundings, video_id + '.npy')

    features = np.load(feat_path)
    features_new = np.zeros(features.shape)
    relation = np.zeros((features.shape[0], len(_store), features.shape[-1]))
    objects = np.zeros((features.shape[0], len(_store), 2 * features.shape[-1]))
    boxes = np.load(box_path)
    boxes = boxes[..., 1:]
    boxes_new = np.zeros(boxes.shape)
    print(boxes_new.shape)
    assert boxes_new.shape == (20, 5, 4), '========box size is wrong!========'

    length = boxes.shape[0]
    n_obj = boxes.shape[1]
    features_new[0, ...] = features[0, ...]
    relation[0, ...] = features_new[0, _relation_id, ...]
    objects[0, ...] = np.stack([features_new[0, pair, ...].reshape(-1) for pair in _store], axis=0)
    boxes_new[0, ...] = boxes[0, ...]

    for i in range(1, length):
        prev_feat = features_new[i - 1, -n_obj:, ...]
        cur_feat = features[i, -n_obj:, ...]
        cos_matrix = cosine_similar(prev_feat, cur_feat)

        prev_box = boxes_new[i - 1, ...]
        cur_box = boxes[i, ...]
        iou_matrix = iou(prev_box, cur_box)
        
        similar_matrix = get_similar_mat(cos_matrix, iou_matrix, lambda_value)
        mapping = get_match(similar_matrix)
        
        for j in range(len(mapping)):
            boxes_new[i, j, ...] = boxes[i, mapping[j], ...]
            features_new[i, j, ...] = features[i, mapping[j], ...]
            features_new[i, n_obj ** 2 + j, ...] = features[i, n_obj ** 2 + mapping[j], ...]

        for j in range(len(mapping)):
            for k in range(len(mapping)):
                if j == k: continue
                map_j = mapping[j]
                map_k = mapping[k]
                features_new[i, n_obj + _mat_mapping[j, k], ...] = features[i, n_obj + _mat_mapping[map_j, map_k], ...]
        
        relation[i, ...] = features_new[i, _relation_id, ...]
        objects[i, ...] = np.stack([features_new[i, pair, ...].reshape(-1) for pair in _store], axis=0)
        
    assert relation.shape == (20, 10, 1024) and objects.shape == (20, 10, 2048), 'rearrange failed!'
    print('done rearrange, relation.shape == {rela} objects.shape == {obj}'.format(rela=relation.shape, obj=objects.shape))
    # np.save(rearranged_feat_path, features_new)
    np.save(relation_feat_path, relation)
    np.save(object_feat_path, objects)
    # np.save(rearranged_box_path, boxes_new)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--now_msvd', action='store_true')
    parser.add_argument('--msdn_features', type=str, default='/home/hanhuaye/PythonProject/opensource/MSDN/MSRVTT_features/features')
    parser.add_argument('--relation_features', type=str, default='/home/hanhuaye/PythonProject/detail-captioning/mydata/relation_features')
    parser.add_argument('--object_features', type=str, default='/home/hanhuaye/PythonProject/detail-captioning/mydata/object_features')
    parser.add_argument('--box_boundings', type=str, default='/home/hanhuaye/PythonProject/opensource/MSDN/MSRVTT_boxes/boxes')
    parser.add_argument('--rearranged_box_boundings', type=str, default='/home/hanhuaye/PythonProject/opensource/MSDN/MSRVTT_boxes/rearranged_boxes')
    parser.add_argument('--vid_dict_path', type=str, default='/home/hanhuaye/PythonProject/detail-captioning/mydata/msvd_data/vid_dict.pkl')
    parser.add_argument('--lambda_value', type=float, default=0.5)
    parser.add_argument('--range', type=int, default=10000)
    parser.add_argument('--n_obj', type=int, default=5)
    args = parser.parse_args()
    n_obj = args.n_obj
    now_msvd = args.now_msvd
    vid_dict_path = args.vid_dict_path

    global _x, _y, _s, _permutations, _mat_mapping, _store, _relation_id
    _store, _relation_id = [], []
    _y, _x = np.meshgrid(range(n_obj), range(n_obj))
    _x, _y = _x.flatten(), _y.flatten()
    _s = [i for i in range(0, n_obj)]
    _permutations = list(itertools.permutations(_s, n_obj))
    _mat_mapping = np.zeros((n_obj, n_obj), dtype=int)
    cnt = 0
    for i in range(n_obj):
        for j in range(n_obj):
            if i == j: 
                _mat_mapping[i][j] = -1
            else:
                _mat_mapping[i][j] = cnt
                cnt += 1
    assert cnt == n_obj ** 2 - n_obj, 'something must goes wrong!'
    
    for i in range(n_obj):
        for j in range(n_obj):
            if i >= j: continue
            _store.append((i, j))

    cnt = n_obj
    for i in range(n_obj):
        for j in range(n_obj):
            if i == j: continue
            if i < j: _relation_id.append(cnt)
            cnt += 1

    if not now_msvd:
        for i in range(args.range):
            rearrange(args, i)
    else:
        with open(vid_dict_path, 'rb') as f:
            vid_dict = pickle.load(f)
            video_name_list = vid_dict.keys()

            for video_name in video_name_list:
                rearrange(args, video_name)


    print("==========================rearrange_object DONE============================")
