#!/usr/bin/env python
# coding=utf-8
import torch
import torch.nn as nn
import numpy as np

class Linear_layer(nn.Module):
    def __init__(self, seed, drop_prob, input_size, output_size, length, is_1d):
        super(Linear_layer, self).__init__()
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        self.linear = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.BatchNorm1d(length) if is_1d else nn.BatchNorm2d(length),
            nn.ReLU(True),
            nn.Dropout(drop_prob)
        )

    def forward(self, features):
        return self.linear(features)


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        seed = args.seed
        drop_prob = args.drop_prob
        n_objs = args.n_objs
        res2d_size = args.res2d_size
        i3d_size = args.i3d_size
        relation_size = args.relation_size
        object_size = args.object_size
        rnn_size = args.rnn_size
        length = args.length
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        self.res_to_rnn = Linear_layer(seed, drop_prob, res2d_size, rnn_size, length, True)
        self.i3d_to_rnn = Linear_layer(seed, drop_prob, i3d_size, rnn_size, length, True)
        self.relation_to_rnn = Linear_layer(seed, drop_prob, relation_size, rnn_size, length, False)
        self.object_to_rnn = Linear_layer(seed, drop_prob, object_size, 2 * rnn_size, length, False)

    def forward(self, res2d_feats, i3d_feats, relation_feats, object_feats, frame_mask, i3d_mask):
        """
        Shapes:
            res2d_feats: (batch, 20, 2048)
            i3d_feats: (batch, 20, 1024)
            relation_feats: (batch, 20, 10, 1024)
            object_feats: (batch, 20, 10, 2048)
            frame_mask: (batch, 20)
            i3d_mask: (batch, 20)
        """
        frame_mask = frame_mask.float()
        i3d_mask = i3d_mask.float()

        return self.res_to_rnn(res2d_feats) * frame_mask.unsqueeze(-1), \
                self.i3d_to_rnn(i3d_feats) * i3d_mask.unsqueeze(-1), \
                self.relation_to_rnn(relation_feats) * frame_mask.unsqueeze(-1).unsqueeze(-1), \
                self.object_to_rnn(object_feats) * frame_mask.unsqueeze(-1).unsqueeze(-1)


class Encoder_trans(nn.Module):
    def __init__(self, args):
        super(Encoder_trans, self).__init__()
        seed = args.seed
        self.drop_prob = args.drop_prob
        self.d_model = args.d_model
        self.length = args.length
        self.res2d_size = args.res2d_size
        self.i3d_size = args.i3d_size
        self.relation_size = args.relation_size
        self.object_size = args.object_size

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        
        self.res_convert = Linear_layer(seed, self.drop_prob, self.res2d_size, self.d_model, self.length, True)
        self.i3d_convert = Linear_layer(seed, self.drop_prob, self.i3d_size, self.d_model, self.length, True)
        self.relation_convert = Linear_layer(seed, self.drop_prob, self.relation_size, self.d_model, self.length, False)
        self.object_convert = Linear_layer(seed, self.drop_prob, self.object_size, self.d_model, self.length, False)


    def forward(self, res2d_feats, i3d_feats, relation_feats, object_feats):
        """
        Pickup static/dynamic frames up to 20. Abandoned mask mechanism.

        Args:
            res2d_feats
            i3d_feats
            relation_feats
            object_feats

        Shapes:
            res2d_feats.shape == (batch, 20, 2048)
            i3d_feats.shape == (batch, 20, 1024)
            relation_feats.shape == (batch, 20, 10, 1024)
            object_feats.shape == (batch, 20, 10, 2048)
            ====================================
            vis_feats0.shape == (batch, 20, 512 * 2)
            vis_feats1.shape == (batch, 20, 512 * 2)
        """
        batch_size = res2d_feats.shape[0]
        res2d_feats = self.res_convert(res2d_feats) # ->(batch, 20, 512)
        i3d_feats = self.i3d_convert(i3d_feats) # ->(batch, 20, 512)
        relation_feats = self.relation_convert(relation_feats) # ->(batch, 20, 10, 512)
        object_feats = self.object_convert(object_feats) # ->(batch, 20, 10, 512)

        relation_feats = relation_feats.view(batch_size, -1, self.d_model).contiguous() # ->(batch, 200, 512)
        object_feats = object_feats.view(batch_size, -1, self.d_model).contiguous() # ->(batch, 200, 512)

        global_feats = torch.cat([res2d_feats, i3d_feats], dim=-1)
        detail_feats = torch.cat([relation_feats, object_feats], dim=-1)

        return global_feats, detail_feats

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--drop_prob', type=float, default=0.5)
    parser.add_argument('--n_objs', type=int, default=5)
    parser.add_argument('--res2d_size', type=int, default=2048)
    parser.add_argument('--i3d_size', type=int, default=1024)
    parser.add_argument('--relation_size', type=int, default=1024)
    parser.add_argument('--object_size', type=int, default=2048)
    parser.add_argument('--rnn_size', type=int, default=512)
    parser.add_argument('--length', type=int, default=20)
    args = parser.parse_args()
    
    temp_encoder = Encoder(args)

    res2d_feats = torch.ones((64, 20, 2048))
    i3d_feats = torch.ones((64, 20, 1024))
    relation_feats = torch.ones((64, 20, 10, 1024))
    object_feats = torch.ones((64, 20, 10, 2048))
    frame_mask = torch.ones((64, 20))
    i3d_mask = torch.ones((64, 20))

    out_res2d, out_i3d, out_relation, out_object = temp_encoder(res2d_feats, i3d_feats, relation_feats, object_feats, 
                                                                frame_mask, i3d_mask)
    print(out_res2d.shape)
    print(out_i3d.shape)
    print(out_relation.shape)
    print(out_object.shape)
