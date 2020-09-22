#!/usr/bin/env python
# coding=utf-8
import torch
import torch.nn as nn
from .lstm_cell import Three_inputs_lstmcell

class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        seed = args.seed
        drop_prob = args.drop_prob
        rnn_size = args.rnn_size
        scene_size = rnn_size * 2
        relation_size = rnn_size * 3
        word_size = rnn_size

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        self.INF = 100000.
        self.n_objs = args.n_objs
        self.h_to_a = nn.Linear(rnn_size, rnn_size)

        self.res2d_to_a = nn.Linear(rnn_size, rnn_size)
        self.res_to_e = nn.Linear(rnn_size, 1)
        
        self.i3d_to_a = nn.Linear(rnn_size, rnn_size)
        self.i3d_to_e = nn.Linear(rnn_size, 1)
        
        self.relation_to_a = nn.Linear(rnn_size, rnn_size)
        self.relation_to_e = nn.Linear(rnn_size, 1)

        self.lstmcell = Three_inputs_lstmcell(scene_size, relation_size, word_size, rnn_size, drop_prob)


    def forward(self, res2d, i3d, relation_feats, object_feats, word, state, res_mask, i3d_mask,  word_mask):
        last_state_h, last_state_c = state
        last_state_h, last_state_c = last_state_h[0], last_state_c[0]
        res_mask = ~res_mask
        i3d_mask = ~i3d_mask

        res_mask = res_mask.float() * -self.INF
        i3d_mask = i3d_mask.float() * -self.INF
        res_mask, i3d_mask = res_mask.float(), i3d_mask.float()

        """
        process res2d_feats and Ui3d_feats
        """
        Wh = self.h_to_a(last_state_h.unsqueeze(1))
        Ures = self.res2d_to_a(res2d)
        Ui3d = self.i3d_to_a(i3d)
        
        # res2d_feats attention
        e_res = self.res_to_e(torch.tanh(Wh + Ures)).squeeze(-1) + res_mask
        beta_res = torch.softmax(e_res, dim=1).unsqueeze(-1)
        attention_res2d = torch.sum(res2d * beta_res, dim=1)

        # i3d_feats attention
        e_i3d = self.i3d_to_e(torch.tanh(Wh + Ui3d)).squeeze(-1) + i3d_mask
        beta_i3d = torch.softmax(e_i3d, dim=1).unsqueeze(-1)
        attention_i3d = torch.sum(i3d * beta_i3d, dim=1)

        """
        process object_feats and relation_feats
        """
        # relation_feats attention
        relation_feats = torch.sum(relation_feats * beta_res.unsqueeze(-1), dim=1)
        Urelation = self.relation_to_a(relation_feats)
        e_relation = self.relation_to_e(torch.tanh(Wh + Urelation)).squeeze(-1)
        beta_relation = torch.softmax(e_relation, dim=1).unsqueeze(-1)
        attention_relation = torch.sum(relation_feats * beta_relation, dim=1)

        # object_feats attention
        object_feats = torch.sum(object_feats * beta_res.unsqueeze(-1), dim=1)
        attention_object = torch.sum(object_feats * beta_relation, dim=1)

        """
        input feats in model
        """
        scene_feats = torch.cat([attention_res2d, attention_i3d], dim=-1)
        relation_feats = torch.cat([attention_relation, attention_object], dim=-1)
        word_feats = word

        output, state = self.lstmcell(scene_feats, relation_feats, word_feats, state, word_mask)
        return output, state

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--drop_prob', type=float, default=0.5)
    parser.add_argument('--n_objs', type=int, default=5)
    parser.add_argument('--rnn_size', type=int, default=512)
    args = parser.parse_args()

    batch = 64
    seq_len = 20
    rnn = 512
    temp_decoder = Decoder(args)
    res2d = torch.ones((batch, seq_len, rnn))
    i3d = torch.ones((batch, seq_len, rnn))
    relation_feats = torch.ones((batch, seq_len, 10, rnn))
    object_feats = torch.ones((batch, seq_len, 10, rnn * 2))
    word = torch.ones((batch, rnn))
    state = (torch.ones(1, batch, rnn), torch.ones(1, batch, rnn))
    res_mask = torch.ones(batch, seq_len).bool()
    i3d_mask = torch.ones(batch, seq_len).bool()
    word_mask = torch.ones(batch, 1)
    
    output, state = temp_decoder(res2d, i3d, relation_feats, object_feats, word, state, res_mask, i3d_mask, word_mask)
    print(output.shape)
    print(state[0].shape)
    print(state[1].shape)
