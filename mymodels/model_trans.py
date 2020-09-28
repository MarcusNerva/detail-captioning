#!/usr/bin/env python
# coding=utf-8
import torch
import torch.nn as nn
from .transformer import Transformer
from .encoder import Encoder_trans

class CaptionModel_trans(nn.Module):
    """
    This CaptionModel is composed of Encoder_trans and Transformer, 
    and the former is encoder and the latter is decoder.
    """
    def __init__(self, args):
        seed = args.seed
        self.beam_size = args.beam_size
        self.max_seq_len = args.seq_length
        self.bos_idx = args.bos_idx
        self.eos_idx = args.eos_idx

        self.encoder = Encoder_trans(args)
        self.transformer = Transformer(args)
        self.args = args

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    def forward(self, res2d_feats, i3d_feats, relation_feats, object_feats, word_seq):
        vis_feats = self.encoder(res2d_feats, i3d_feats, relation_feats, object_feats)
        ret = self.transformer(vis_feats, word_seq)
        return ret

    def generate_sentence(self, res2d_feats, i3d_feats, relation_feats, object_feats):
        vis_feats = self.encoder(res2d_feats, i3d_feats, relation_feats, object_feats)
        ret = self.transformer.generate_sentence(vis_feats, self.beam_size, self.max_seq_len, self.bos_idx, self.eos_idx)
        return ret
    