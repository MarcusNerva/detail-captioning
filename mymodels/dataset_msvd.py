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


class DatasetMSVD(Dataset):
    """
    This dataset class comprises 3 kinds of mode, namely, training, validation, testing.

    1,200 videos for training, 100 for validation and 670 for testing.
    I split this MSVD dataset in these 3 part according to they rank in csv file.

    Validation mode and testing mode is similar,
    where the model we valid/test here needs raw captions and visual features.
    In training mode, the model needs numericalized captions and visual features, however.
    """

    def __init__(self, mode, args):
        """

        Args:
            mode: 'train', 'valid', 'test'.
            args:  the global settings defined in mycfgs/cfgs.py
        """
        super(DatasetMSVD, self).__init__()
        self.mode = mode
        self.args = args
        self.pad_idx = None
        self.bos_idx = None
        self.eos_idx = None
        self.unk_idx = None
        self.n_vocab = None
        self.stoi = None
        self.itos = None

        self.res2d_dir = os.path.join(args.msvd_data_dir, args.res2d_subdir)
        self.i3d_dir = os.path.join(args.msvd_data_dir, args.i3d_subdir)
        self.relation_dir = os.path.join(args.msvd_data_dir, args.relation_subdir)
        self.object_dir = os.path.join(args.msvd_data_dir, args.object_subdir)

        torchtext_path = os.path.join(args.msvd_data_dir, args.torchtext_subpath)
        seq_dict_path = os.path.join(args.msvd_data_dir, args.seq_dict_subpath)
        numberic_dict_path = os.path.join(args.msvd_data_dir, args.numberic_dict_subpath)
        seq_mask_path = os.path.join(args.msvd_data_dir, args.seq_mask_subpath)
        res2d_mask_path = os.path.join(args.msvd_data_dir, args.res2d_mask_subpath)
        i3d_mask_path = os.path.join(args.msvd_data_dir, args.i3d_mask_subpath)
        vid_dict_path = os.path.join(args.msvd_data_dir, args.vid_dict_subpath)

        if not os.path.exists(torchtext_path) or not os.path.exists(seq_dict_path) \
            or not os.path.exists(numberic_dict_path) or not os.path.exists(seq_mask_path) \
            or not os.path.exists(vid_dict_path):

            print('extracting words.........')
            get_vocab_and_seq(args)
            print('extracting succeed!')

        self.numberic = []
        self.video_name = []
        self.word_mask = []
        self.res2d_feats = {}
        self.i3d_feats = {}
        self.relation_feats = {}
        self.object_feats = {}
        self.res2d_mask = np.load(res2d_mask_path)
        self.i3d_mask = np.load(i3d_mask_path)
        with open(seq_mask_path, 'rb') as f:
            seq_mask = pickle.load(f)
        with open(numberic_dict_path, 'rb') as f:
            numberic_dict = pickle.load(f)
        with open(torchtext_path, 'rb') as f:
            text_proc = pickle.load(f)
        with open(seq_dict_path, 'rb') as f:
            self.seq_dict = pickle.load(f)
        with open(vid_dict_path, 'rb') as f:
            self.vid_dict = pickle.load(f)
        self.video_name_list = self._define_video_name_list()

        for video_name in self.video_name_list:
            res2d_path = os.path.join(self.res2d_dir, video_name + '.npy')
            i3d_path = os.path.join(self.i3d_dir, video_name + '.npy')
            relation_path = os.path.join(self.relation_dir, video_name + '.npy')
            object_path = os.path.join(self.object_dir, video_name + '.npy')

            self.res2d_feats[video_name] = np.load(res2d_path)
            self.i3d_feats[video_name] = np.load(i3d_path)
            self.relation_feats[video_name] = np.load(relation_path)
            self.object_feats[video_name] = np.load(object_path)

            for number_tensor in numberic_dict[video_name]:
                self.numberic.append(number_tensor)
                self.video_name.append(video_name)
            for mask in seq_mask[video_name]:
                self.word_mask.append(mask)

        pad = args.pad
        bos = args.bos
        eos = args.eos
        unk = args.unk
        self.pad_idx = text_proc.vocab.stoi[pad]
        self.bos_idx = text_proc.vocab.stoi[bos]
        self.eos_idx = text_proc.vocab.stoi[eos]
        self.unk_idx = text_proc.vocab.stoi[unk]
        self.n_vocab = len(text_proc.vocab.stoi)
        self.stoi = text_proc.vocab.stoi
        self.itos = text_proc.vocab.itos

    def __getitem__(self, idx):
        video_name = self.video_name_list[idx]
        numb_vid = self.vid_dict[video_name]
        res2d_feat = self.res2d_feats[video_name]
        i3d_feat = self.i3d_feats[video_name]
        relation_feat = self.relation_feats[video_name]
        object_feat = self.object_feats[video_name]

        return res2d_feat, \
                i3d_feat, \
                relation_feat, \
                object_feat, \
                self.res2d_mask[numb_vid], \
                self.i3d_mask[numb_vid], \
                self.numberic[idx], \
                self.word_mask[idx], \
                self.seq_dict[video_name], \
                video_name

    def get_pad_idx(self):
        return self.pad_idx

    def get_bos_idx(self):
        return self.bos_idx

    def get_eos_idx(self):
        return self.eos_idx

    def get_unk_idx(self):
        return self.unk_idx

    def get_n_vocab(self):
        return self.n_vocab

    def __len__(self):
        return 20 * len(self.data_range)

    def _define_video_name_list(self):
        if self.mode not in ['train', 'valid', 'test']:
            raise NotImplementedError

        video_name_list = list(self.vid_dict.keys())
        if self.mode == 'train':
            return video_name_list[:1200]
        if self.mode == 'valid':
            return video_name_list[1200:1300]
        return video_name_list[1300:]


def msvd_collate_fn(batch):
    res2d, i3d, relation, object_, res2d_mask, i3d_mask, numberic, mask, seq, video_name = zip(*batch)

    res2d = np.stack(res2d, axis=0)
    i3d = np.stack(i3d, axis=0)
    relation = np.stack(relation, axis=0)
    object_ = np.stack(object_, axis=0)
    res2d_mask = np.stack(res2d_mask, axis=0)
    i3d_mask = np.stack(i3d_mask, axis=0)
    numberic = np.stack(numberic, axis=0)
    mask = np.stack(mask, axis=0)
    seq = [s for s in seq]
    video_name = [v for v in video_name]

    res2d = torch.from_numpy(res2d)
    i3d = torch.from_numpy(i3d)
    relation = torch.from_numpy(relation)
    object_ = torch.from_numpy(object_)
    res2d_mask = torch.from_numpy(res2d_mask).bool()
    i3d_mask = torch.from_numpy(i3d_mask).bool()
    numberic = torch.from_numpy(numberic)
    mask = torch.from_numpy(mask)

    return res2d, i3d, relation, object_, res2d_mask, i3d_mask, numberic, mask, seq, video_name

if __name__ == '__main__':
    args = get_total_settings()
    get_vocab_and_seq(args)