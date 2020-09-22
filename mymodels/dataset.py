#!/usr/bin/env python
# coding=utf-8
import torch
import torchtext
from torch.utils.data import Dataset
import numpy as np
import os
import glob
import json
import pickle
from collections import defaultdict
import sys
sys.path.append('../')
from myscripts.make_datastore import make_datastore


def get_vocab_and_seq(args):
    """
    This function is aimming to process captions in json file. And save these data on disk through serialization.
    Args:
        args: the global settings which are defined in tools/settings.py
    Savings:
        text_proc: torchtext.data.Field instance. Return it for utilizing it's vocab
        seq_dict: {video_id -> [caption * 20]}
    """

    data_dir = args.data_dir
    bos = args.bos
    eos = args.eos
    pad = args.pad
    max_length = args.seq_length
    
    json_path = args.json_path
    torchtext_path = os.path.join(data_dir, 'torchtext.pkl')
    seq_dict_path = os.path.join(data_dir, 'seq_dict.pkl')
    seq_mask_path = os.path.join(data_dir, 'seq_mask.pkl')
    numberic_dict_path = os.path.join(data_dir, 'numberic_dict.pkl')

    text_proc = torchtext.data.Field(sequential=True, init_token=bos, eos_token=eos, pad_token=eos, 
                                    tokenize='spacy', lower=True, batch_first=True, fix_length=max_length)
    seq_dict = defaultdict(list)
    seq_mask = {}
    numberic_dict = {}
    seqs_store = []

    with open(json_path, 'r') as f:
        json_file = json.load(f)
        sentences = json_file['sentences']
        for temp_dict in sentences:
            video_id = temp_dict['video_id']
            caption = temp_dict['caption'].strip()
            seq_dict[video_id].append(caption)
            seqs_store.append(caption)

    seqs_store = list(map(text_proc.preprocess, seqs_store))
    text_proc.build_vocab(seqs_store, min_freq=1)
    temp_dict = text_proc.vocab.stoi
    temp_list = text_proc.vocab.itos
    
    temp_dict['<unk>'], temp_dict['<eos>'] = 1, 0
    temp_list[0], temp_list[1] = temp_list[1], temp_list[0]

    for video_id in seq_dict:
        temp_seq_mask = np.zeros((20, max_length))
        sentence_list = seq_dict[video_id]
        assert len(sentence_list) == 20, 'something goes wrong with seq_dict in dataset.py'
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


class DatasetMSRVTT(Dataset):
    """
    This dataset class comprises 3 kinds of mode, namely, training, validation, testing.
    
    I utilize first 6513 videos as train_dataset, that is, video_id in [0, 6513).
    Subsequently, 497 videos in the middle is utilized as valid_dataset, that is, video_id in [6513, 7010).
    Finally, the rest videos is utilized as test_dataset, that is, video_id in [7010, 10000).
    
    Validation mode and testing mode is similar, where the model we valid/test here needs raw captions and visual features.
    In training mode, the model needs numericalized captions and visual features, however.
    """

    def __init__(self, mode, args):
        """
        Args:
            mode: 'train', 'valid', 'test'.
            args: the global settings defined in tools/settings.py
        """
        super(DatasetMSRVTT, self).__init__()
        self.mode = mode
        self.args = args
        self.pad_idx = None
        self.bos_idx = None
        self.eos_idx = None
        self.n_vocab = None
        self.stoi = None
        self.itos = None 
        self.data_range = self._define_data_range()

        torchtext_path = args.torchtext_path
        seq_dict_path = args.seq_dict_path
        numberic_dict_path = args.numberic_dict_path
        datastore_dir = None

        if mode == 'train':
            datastore_dir = args.train_datastore_dir
        elif mode == 'valid':
            datastore_dir = args.valid_datastore_dir
        else:
            datastore_dir = args.test_datastore_dir

        res2d_path = os.path.join(datastore_dir, 'res2d.pkl')
        i3d_path = os.path.join(datastore_dir, 'i3d.pkl')
        relation_path = os.path.join(datastore_dir, 'relation.pkl')
        object_path = os.path.join(datastore_dir, 'object.pkl')
        numberic_path = os.path.join(datastore_dir, 'numberic.pkl')
        video_id_path = os.path.join(datastore_dir, 'video_id.pkl')
        word_mask_path = os.path.join(datastore_dir, 'word_mask.pkl')

        if not os.path.exists(torchtext_path) or not os.path.exists(seq_dict_path) or not os.path.exists(numberic_dict_path):
            print('extracting words.........')
            get_vocab_and_seq(args)
            print('extracting succeed!')

        pkl_list = glob.glob(os.path.join(datastore_dir, '*.pkl'))
        if(len(pkl_list) < 7):
            print('constructing data_store.......')
            make_datastore(args, mode, self.data_range)
            print('constructing finished!')

        with open(torchtext_path, 'rb') as f:
            text_proc = pickle.load(f)
        with open(res2d_path, 'rb') as f:
            self.res2d_dict = pickle.load(f)
        with open(i3d_path, 'rb') as f:
            self.i3d_dict = pickle.load(f)
        with open(relation_path, 'rb') as f:
            self.relation_dict = pickle.load(f)
        with open(object_path, 'rb') as f:
            self.object_dict = pickle.load(f)
        with open(numberic_path, 'rb') as f:
            self.numberic = pickle.load(f)
        with open(video_id_path, 'rb') as f:
            self.video_id = pickle.load(f)
        with open(word_mask_path, 'rb') as f:
            self.word_mask = pickle.load(f)

        self.res2d_mask = np.load(res2d_mask_path)
        self.i3d_mask = np.load(i3d_mask_path)

            
        pad = args.pad
        bos = args.bos
        eos = args.eos
        self.pad_idx = text_proc.vocab.stoi[pad]
        self.bos_idx = text_proc.vocab.stoi[bos]
        self.eos_idx = text_proc.vocab.stoi[eos]
        self.n_vocab = len(text_proc.vocab.stoi)
        self.stoi = text_proc.vocab.stoi
        self.itos = text_proc.vocab.itos


    def __getitem__(self, idx):
        vid = self.video_id[idx]
        numb_vid = int(vid[5:])

        return self.res2d_dict[vid], \
                self.i3d_dict[vid], \
                self.relation_dict[vid], \
                self.object_dict[vid], \
                self.res2d_mask[numb_vid], \
                self.i3d_mask[numb_vid], \
                self.numberic[idx], \
                self.mask[idx], \
                self.seq_dict[vid]

    def get_pad_idx(self):
        return self.pad_idx

    def get_bos_idx(self):
        return self.eos_idx

    def get_eos_idx(self):
        return self.eos_idx

    def get_n_vocab(self):
        return self.n_vocab

    def __len__(self):
        return 20 * len(self.data_range)

    def _define_data_range(self):

        if self.mode not in ['train', 'valid', 'test']:
            raise NotImplementedError

        ret = ['video' + str(i) for i in range(10000)]
        if self.mode == 'train':
            return ret[:6513]
        elif self.mode == 'valid':
            return ret[6513:7010]
        else:
            return ret[7010:]
    
    def get_stoi(self):
        return self.stoi

    def get_itos(self):
        return self.itos

def collate_fn(batch):
    res2d, i3d, relation, object_, res2d_mask, i3d_mask, numberic, mask, seq = zip(*batch)
    
    res2d = np.stack(res2d, axis=0)
    i3d = np.stack(i3d, axis=0)
    relation = np.stack(relation, axis=0)
    object_ = np.stack(object_, axis=0)
    res2d_mask = np.stack(res2d_mask, axis=0)
    i3d_mask = np.stack(i3d_mask, axis=0)
    numberic = np.stack(numberic, axis=0)
    mask = np.stack(mask, axis=0)
    seq = [s for s in seq]

    res2d = torch.from_numpy(res2d)
    i3d = torch.from_numpy(i3d)
    relation = torch.from_numpy(relation)
    object_ = torch.from_numpy(object_)
    res2d_mask = torch.from_numpy(res2d_mask)
    i3d_mask = torch.from_numpy(i3d_mask)
    numberic = torch.from_numpy(numberic)
    mask = torch.from_numpy(mask)

    return res2d, i3d, relation, object_, res2d_mask, i3d_mask, numberic, mask, seq
