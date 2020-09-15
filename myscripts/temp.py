#!/usr/bin/env python
# coding=utf-8
import torch
import numpy as np
import itertools
from rearrange_object import softmax

n_obj = 5
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
    print(iou)
    iou = softmax(iou)    
    return iou

if __name__ == '__main__':
    prev = np.asarray([[1, 1, 2, 2], [0, 0, 1, 1], [0, 0, 2, 2], [0, 0, 3, 3], [1, 0, 4, 3]])
    cur = np.asarray([[1, 1, 2, 2], [1, 1, 2, 2], [1, 1, 3, 3], [1, 1, 2, 2], [0, 2, 5, 4]])

    out = iou(prev, cur)
    # print(out)
