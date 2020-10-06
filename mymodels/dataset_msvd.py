import torch
import torchtext
from torch.utils.data import Dataset
import numpy as np
import os
import glob
import pickle
import csv
from collections import defaultdict, OrderedDict
import re
import sys
sys.path.append('../')
from mycfgs import get_total_settings

def get_vocab_and_seq(args):
    """
    This function is aimming to process captions in csv file. And save these data on disk through serialization.
    Args:
        args: the global settings which are defined in mycfgs/cfgs.py
    Savings:
        text_proc: torchtext.data.Field instance. Return it for utilizing it's vocab
        seq_dict: {video_id->[caption * 40(roughly)]}
    """

    data_dir = args.msvd_data_dir
    videos_dir = args.msvd_videos_dir
    csv_subpath = args.csv_subpath
    torchtext_subpath = args.torchtext_subpath
    seq_dict_subpath = args.seq_dict_subpath
    seq_mask_subpath = args.seq_mask_subpath
    numberic_dict_subpath = args.numberic_dict_subpath
    vid_dict_subpath = args.vid_dict_subpath
    bos = args.bos
    eos = args.eos
    pad = args.pad
    unk = args.unk
    max_length = args.seq_length

    csv_path = os.path.join(data_dir, csv_subpath)
    torchtext_path = os.path.join(data_dir, torchtext_subpath)
    seq_dict_path = os.path.join(data_dir, seq_dict_subpath)
    seq_mask_path = os.path.join(data_dir, seq_mask_subpath)
    numberic_dict_path = os.path.join(data_dir, numberic_dict_subpath)
    vid_dict_path = os.path.join(data_dir, vid_dict_subpath)

    video_list = glob.glob(os.path.join(videos_dir, '*.avi'))
    video_name_list = [video_list[i].split('/')[-1].split('.')[0] for i in range(len(video_list))]
    text_proc = torchtext.data.Field(sequential=True, init_token=bos, eos_token=eos, pad_token=eos,
                                     tokenize='spacy', lower=True, batch_first=True, fix_length=max_length)
    seq_dict = OrderedDict()
    seq_mask = {}
    numberic_dict = {}
    seqs_store = []
    vid_dict = {}

    with open(csv_path, 'r') as f:
        csv_file = csv.reader(f)
        for line in csv_file:
            if len(line) == 0: continue
            language = line[6]
            if language != 'English': continue
            video_id = line[0] + '_' + str(line[1]) + '_' + str(line[2])
            if video_id not in video_name_list: continue

            caption = line[7].strip()
            caption = re.sub(r'[^\w\s]', '', caption)
            if video_id not in seq_dict.keys():
                seq_dict[video_id] = []
            seq_dict[video_id].append(caption)
            seqs_store.append(caption)

    seq_dict_keys = list(seq_dict.keys())
    seqs_store = list(map(text_proc.preprocess, seqs_store))
    text_proc.build_vocab(seqs_store, min_freq=1)
    temp_dict = text_proc.vocab.stoi
    temp_list = text_proc.vocab.itos

    temp_dict[unk], temp_dict[eos] = 1, 0
    temp_list[0], temp_list[1] = temp_list[1], temp_list[0]

    for i, (key) in enumerate(seq_dict_keys):
        vid_dict[key] = i

    for video_id in seq_dict:
        sentence_list = seq_dict[video_id]
        sentence_numb = len(sentence_list)
        temp_seq_mask = np.zeros((sentence_numb, max_length))
        discrete_sent_list = list(map(text_proc.preprocess, sentence_list))
        temp_list = []
        for i, (single_list) in enumerate(discrete_sent_list):
            temp_seq_mask[i, :len(single_list) + 1] = 1
            single_sent = ''
            for word in single_list:
                single_sent += word + ' '
            single_sent = single_sent.strip()
            temp_list.append(single_sent)
        seq_dict[video_id] = temp_list
        seq_mask[video_id] = temp_seq_mask

    for video_id in seq_dict:
        sentence_list = seq_dict[video_id]
        discrete_sent_list = list(map(text_proc.preprocess, sentence_list))
        numberic_dict[video_id] = text_proc.numericalize(text_proc.pad(discrete_sent_list)).cpu().numpy()

    with open(torchtext_path, 'wb') as f:
        pickle.dump(text_proc, f)
    with open(seq_dict_path, 'wb') as f:
        pickle.dump(seq_dict, f)
    with open(numberic_dict_path, 'wb') as f:
        pickle.dump(numberic_dict, f)
    with open(seq_mask_path, 'wb') as f:
        pickle.dump(seq_mask, f)
    with open(vid_dict_path, 'wb') as f:
        pickle.dump(vid_dict, f)

if __name__ == '__main__':
    args = get_total_settings()
    get_vocab_and_seq(args)