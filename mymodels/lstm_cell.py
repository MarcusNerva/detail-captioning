#!/usr/bin/env python
# coding=utf-8
import torch
import torch.nn as nn

class Two_inputs_lstmcell(nn.Module):
    def __init__(self, scene_size, word_size, rnn_size, drop_prob):
        super(Two_inputs_lstmcell, self).__init__()
        assert scene_size == 2 * rnn_size, 'scene_size or rnn_size must go wrong!'
        assert word_size == rnn_size, 'word_size or rnn_size must go wrong!'

        self.rnn_size = rnn_size
        self.scene_to_hidden = nn.Linear(scene_size, rnn_size * 4)
        self.word_to_hidden = nn.Linear(word_size, rnn_size * 4)
        self.hidden_to_hidden = nn.Linear(rnn_size, rnn_size * 4)
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, scene_feats, word, state, word_mask=None):
        last_state_h, last_state_c = state
        last_state_h, last_state_c = last_state_h[0], last_state_c[0]

        all_input_sums = self.scene_to_hidden(scene_feats) + \
                         self.word_to_hidden(word) + \
                         self.hidden_to_hidden(last_state_h)

        sigmoid_chunk = all_input_sums.narrow(dim=1, start=0, length=3 * self.rnn_size)
        sigmoid_chunk = torch.sigmoid(sigmoid_chunk)

        tanh_chunk = all_input_sums.narrow(dim=1, start=3 * self.rnn_size, length=self.rnn_size)
        new_info = torch.tanh(tanh_chunk)

        in_gate = sigmoid_chunk.narrow(dim=1, start=0, length=self.rnn_size)
        forget_gate = sigmoid_chunk.narrow(dim=1, start=self.rnn_size, length=self.rnn_size)
        out_gate = sigmoid_chunk.narrow(dim=1, start=self.rnn_size * 2, length=self.rnn_size)

        state_c = forget_gate * last_state_c + in_gate * new_info
        if word_mask is not None:
            state_c = state_c * word_mask + last_state_c * (1. - word_mask)

        state_h = out_gate * torch.tanh(state_c)
        if word_mask is not None:
            state_h = state_h * word_mask + last_state_h * (1. - word_mask)

        if self.dropout is not None:
            state_h = self.dropout(state_h)

        output = state_h
        return output, (state_h.unsqueeze(0), state_c.unsqueeze(0))


class Three_inputs_lstmcell(nn.Module):
    def __init__(self, scene_size, relation_size, word_size, rnn_size, drop_prob):
        super(Three_inputs_lstmcell, self).__init__()
        assert scene_size == 2 * rnn_size, 'scene_size or rnn_size must go wrong!'
        assert relation_size == 3 * rnn_size, 'relation_size or rnn_size must go wrong!'
        assert word_size == rnn_size, 'word_size or rnn_size must go wrong!'

        self.rnn_size = rnn_size
        self.scene_to_hidden = nn.Linear(scene_size, rnn_size * 4)
        self.relation_to_hidden = nn.Linear(relation_size, rnn_size * 4)
        self.word_to_hidden = nn.Linear(word_size, rnn_size * 4)
        self.hidden_to_hidden = nn.Linear(rnn_size, rnn_size * 4)
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, scene_feats, relation_feats, word, state, word_mask=None):
        last_state_h, last_state_c = state
        last_state_h, last_state_c = last_state_h[0], last_state_c[0]

        all_input_sums = self.scene_to_hidden(scene_feats) +\
                self.relation_to_hidden(relation_feats) +\
                self.word_to_hidden(word) +\
                self.hidden_to_hidden(last_state_h)
        
        sigmoid_chunk = all_input_sums.narrow(dim=1, start=0, length=3 * self.rnn_size)
        sigmoid_chunk = torch.sigmoid(sigmoid_chunk)

        tanh_chunk = all_input_sums.narrow(dim=1, start=3 * self.rnn_size, length=self.rnn_size)
        new_info = torch.tanh(tanh_chunk)
        
        in_gate = sigmoid_chunk.narrow(dim=1, start=0, length=self.rnn_size)
        forget_gate = sigmoid_chunk.narrow(dim=1, start=self.rnn_size, length=self.rnn_size)
        out_gate = sigmoid_chunk.narrow(dim=1, start=self.rnn_size * 2, length=self.rnn_size)
        
        state_c = forget_gate * last_state_c + in_gate * new_info
        if word_mask is not None:
            state_c = state_c * word_mask + last_state_c * (1. - word_mask)
        
        state_h = out_gate * torch.tanh(state_c)
        if word_mask is not None:
            state_h = state_h * word_mask + last_state_h * (1. - word_mask)

        if self.dropout is not None:
            state_h = self.dropout(state_h)

        output = state_h
        return output, (state_h.unsqueeze(0), state_c.unsqueeze(0))

