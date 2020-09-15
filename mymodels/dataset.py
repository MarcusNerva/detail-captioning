#!/usr/bin/env python
# coding=utf-8
import torch
import torchtext
from torch.utils.data import Dataset
import numpy as np
import os
import json
import pickle
from collections import defaultdict

def get_vocab_and_seq(args):
    """
    This function is aimming to process captions in json file. And save these data on disk through serialization.
    Args:
        args: the global settings which are defined in tools/settings.py
    Savings:
        text_proc: torchtext.data.Field instance. Return it for utilizing it's vocab
        seq_dict: {video_id -> [caption * 20]}
    """

    data_path = args.data_path
    bos = args.bos
    eos = args.eos
    pad = args.pad
    max_length = args.seq_length
    
    json_path = args.json_path
    torchtext_path = os.path.join(data_path, 'torchtext.pkl')
    seq_dict_path = os.path.join(data_path, 'seq_dict.pkl')
    seq_mask_path = os.path.join(data_path, 'seq_mask.npy')
    numberic_dict_path = os.path.join(data_path, 'numberic_dict.npy')

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
        for i, [single_list] in enumerate(discrete_sent_list):
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
    np.save(numberic_dict_path, numberic_dict)
    np.save(seq_mask_path, seq_mask)

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
        self.res2d = []
        self.i3d = []
        self.relation = []
        self.object = []
        self.numberic = []
        self.mask = []
        self.total_numberic = []
        self.total_sentences = []
        self.video_id = []

        res2d_dir = args.res2d_dir
        i3d_dir = args.i3d_dir
        relation_dir = args.relation_dir
        object_dir = args.object_dir
        res2d_mask_path = args.res2d_mask_path
        i3d_mask_path = args.i3d_mask_path
        seq_mask_path = args.seq_mask_path
        torchtext_path = args.torchtext_path
        seq_dict_path = args.seq_dict_path
        numberic_dict_path = args.numberic_dict_path

        if not os.path.exists(torchtext_path) or not os.path.exists(seq_dict_path) or not os.path.exists(numberic_dict_path):
            print('extracting words.........')
            get_vocab_and_seq(args)
            print('extracting succeed!')

        with open(torchtext_path, 'rb') as f:
            text_proc = pickle.load(f)
        with open(seq_dict_path, 'rb') as f:
            self.seq_dict = pickle.load(f)
        self.numberic_dict = np.load(numberic_dict_path)
        self.res2d_mask = np.load(res2d_mask_path)
        self.i3d_mask = np.load(i3d_mask_path)
        self.seq_mask = np.load(seq_mask_path)


        for video_id in self.data_range:
            vid = int(video_id[5:])
            res2d_path = os.path.join(res2d_dir, video_id + '.npy')
            i3d_path = os.path.join(i3d_dir, video_id + '.npy')
            relation_path = os.path.join(relation_dir, video_id + '.npy')
            object_path = os.path.join(object_dir, video_id + '.npy')

            res2d = np.load(res2d_path)
            i3d = np.load(i3d_path)
            relation = np.load(relation_path)
            object_ = np.load(object_path)

            self.res2d.append(res2d)
            self.i3d.append(i3d)
            self.relation.append(relation)
            self.object.append(object_)
            for number_tensor in self.numberic_dict[video_id]:
                self.numberic.append(number_tensor)
                self.video_id.append(vid)
            for mask in self.seq_mask[video_id]:
                self.mask.append(mask)
            
            
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
        str_vid = 'video' + str(vid)

        return self.res2d[vid], \
                self.i3d[vid], \
                self.relation[vid], \
                self.object[vid], \
                self.res2d_mask[str_vid], \
                self.i3d_mask[str_vid], \
                self.numberic[idx], \
                self.mask[idx], \
                self.seq_dict[str_vid]

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
    
    res2d = torch.stack(res2d, dim=0)
    i3d = torch.stack(i3d, dim=0)
    relation = torch.stack(relation, dim=0)
    object_ = torch.stack(object_, dim=0)
    res2d_mask = torch.stack(res2d_mask, dim=0)
    i3d_mask = torch.stack(i3d_mask, dim=0)
    numberic = torch.stack(numberic, dim=0)
    mask = torch.stack(mask, dim=0)
    seq = [s for s in seq]

    return res2d, i3d, relation, object_, res2d_mask, i3d_mask, numberic, mask, seq
