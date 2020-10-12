#!/usr/bin/env python
# coding=utf-8
from collections import OrderedDict, defaultdict
from .coco_caption.pycocoevalcap.bleu.bleu import Bleu
from .coco_caption.pycocoevalcap.cider.cider import Cider
from .coco_caption.pycocoevalcap.meteor.meteor import Meteor
from .coco_caption.pycocoevalcap.rouge.rouge import Rouge
from torch.utils.data.dataloader import DataLoader
import torch
import os
import pickle

def language_eval(sample_seqs, groundtruth_seqs):
    assert len(sample_seqs) == len(groundtruth_seqs), 'length of sampled seqs is different from that of groundtruth seqs!'

    references, predictions = OrderedDict(), OrderedDict()
    for i in range(len(groundtruth_seqs)):
        references[i] = [groundtruth_seqs[i][j] for j in range(len(groundtruth_seqs[i]))]
    for i in range(len(sample_seqs)):
        predictions[i] = [sample_seqs[i]]

    predictions = {i: predictions[i] for i in range(len(sample_seqs))}
    references = {i: references[i] for i in range(len(groundtruth_seqs))}

    avg_bleu_score, bleu_score = Bleu(4).compute_score(references, predictions)
    print('avg_bleu_score == ', avg_bleu_score)
    avg_cider_score, cider_score = Cider().compute_score(references, predictions)
    print('avg_cider_score == ', avg_cider_score)
    avg_meteor_score, meteor_score = Meteor().compute_score(references, predictions)
    print('avg_meteor_score == ', avg_meteor_score)
    avg_rouge_score, rouge_score = Rouge().compute_score(references, predictions)
    print('avg_rouge_score == ', avg_rouge_score)

    return {'BLEU': avg_bleu_score, 'CIDEr': avg_cider_score, 'METEOR': avg_meteor_score, 'ROUGE': avg_rouge_score}


def decode_idx(seq, itow, eos_idx):
    ret = ''
    length = seq.shape[0]
    for i in range(length):
        if seq[i] == eos_idx: break
        if i > 0: ret += ' '
        ret += itow[seq[i]]
    return ret


@torch.no_grad()
def eval(args, model, dataset, device, collate_fn, situation=None):
    if situation is not None:
        prediction_store_path = os.path.join(args.msvd_data_dir if args.now_msvd else args.data_dir,
                                            situation + '_prediction.pkl')
        predictions_store = defaultdict(list)

    batch_size = args.batch_size
    itow = dataset.get_itos()
    model.eval()
    dataloader = DataLoader(dataset, batch_size, shuffle=True, collate_fn=collate_fn)
    predictions, gts = [], []

    for i, (res2d, i3d, relation, object_, res2d_mask, i3d_mask, numberic, mask, seq, video_name) in enumerate(dataloader):
        res2d = res2d.to(device)
        i3d = i3d.to(device)
        relation = relation.to(device)
        object_ = object_.to(device)
        res2d_mask = res2d_mask.to(device)
        i3d_mask = i3d_mask.to(device)
        numberic = numberic.to(device)
        mask = mask.to(device)
        
        pred, _ = model.sample(res2d, i3d, relation, object_, res2d_mask, i3d_mask)
        pred = pred.cpu().numpy()
        pred = [decode_idx(temp_seq, itow, args.eos_idx) for temp_seq in pred]
        predictions += pred
        seq = [single_seq_list for single_seq_list in seq]
        gts += seq

        if situation is not None:
            batch = res2d.shape[0]
            for i in range(batch):
                video_id = video_name[i]
                temp_pred = pred[i]
                predictions_store[video_id].append(temp_pred)


    model.train()
    language_state = language_eval(predictions, gts)

    if situation is not None:
        with open(prediction_store_path, 'wb') as f:
            pickle.dump(predictions_store, f)

    return language_state

"""
if __name__ == '__main__':
    sample_seqs = ['train traveling down a track in front of a road']
    groundtruth_seqs = [['A train traveling down tracks next to lights',
                       'A blue and silver train next to train station and trees',
                       'A blue train is next to a sidewalk on the rails',
                       'A passenger train pulls into a train station',
                       'A train coming down the tracks arriving at a station']]

    language_eval(sample_seqs, groundtruth_seqs)
"""
