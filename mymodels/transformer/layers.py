#!/usr/bin/env python
# coding=utf-8
import torch
import torch.nn as nn
from .attention import MultiHeadAttention
from .positionwise_feedforward import PositionwiseFeedForward
from ..encoder import Linear_layer

class EncoderLayer(nn.Module):
    """
    EncoderLayer = self_attention with sublayerconnection + feedforward with sublayerconnection
    """
    def __init__(self, d_model, d_hidden, n_heads, dropout=0.3):
        """
        Args:
            d_model: hidden size of transformer
            d_hidden: feed forward hidden size, usually 4 * d_model
            n_heads: number of heads in multi-head attention
            dropout: dropout rate
        """
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(n_heads=n_heads, d_model=d_model, dropout=dropout)
        self.pos_feedforward = PositionwiseFeedForward(d_model=d_model, d_hidden=d_hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, encoder_input, mask=None):
        encoder_input, encoder_attn = self.self_attention(encoder_input, encoder_input, encoder_input, mask=mask)
        encoder_output = self.pos_feedforward(encoder_input)
        return encoder_output, encoder_attn
        

class DecoderLayer(nn.Module):
    """
    DecoderLayer = self_attention + enc_attention + feedforward
    """
    def __init__(self, d_model, d_hidden, n_heads, dropout=0.3):
        """
        Args:
            d_model: hidden size of transformer
            d_hidden: feed forward hidden size, usually 4 * d_model
            n_heads: number of heads in multi-head attention
            dropout: dropout rate
        """
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(n_heads=n_heads, d_model=d_model, dropout=dropout)

        self.enc_attention_global = MultiHeadAttention(n_heads=n_heads, d_model=d_model, dropout=dropout)
        self.enc_attention_detail = MultiHeadAttention(n_heads=n_heads, d_model=d_model, dropout=dropout)

        self.pos_feedforward_global = PositionwiseFeedForward(d_model=d_model, d_hidden=d_hidden, dropout=dropout)
        self.pos_feedforward_detail = PositionwiseFeedForward(d_model=d_model, d_hidden=d_hidden, dropout=dropout)

        self.linear = Linear_layer(seed=1, drop_prob=dropout, input_size=d_model * 2, output_size=d_model, is_1d=True)


    def forward(self, encoder_output, decoder_input, self_attn_mask=None, enc_attn_mask=None):
        global_feats, detail_feats = encoder_output
        decoder_output, _ = self.self_attention(decoder_input, decoder_input, decoder_input, mask=self_attn_mask)

        decoder_output_global, _ = self.enc_attention_global(decoder_output, global_feats, global_feats, mask=enc_attn_mask)
        decoder_output_detail, _ = self.enc_attention_detail(decoder_output, detail_feats, detail_feats, mask=enc_attn_mask)

        decoder_output = torch.cat([decoder_output_detail, decoder_output_global], dim=-1)
        decoder_output = self.linear(decoder_output)

        return decoder_output, None, None

if __name__ == '__main__':
    d_model = 512
    d_hidden = 2048
    n_heads = 8
    batch_size = 16
    len_frames = 20

    encoder_input = torch.ones(batch_size, len_frames, d_model)
    decoder_output = torch.ones(batch_size, len_frames // 2, d_model)
    encoder_mask = torch.ones(batch_size, 1, len_frames)
    self_attn_mask = (1 - torch.triu(torch.ones(1, len_frames // 2, len_frames // 2), diagonal=1))
    dec_attn_mask = encoder_mask.clone()
    
    temp_encoder = EncoderLayer(d_model=d_model, d_hidden=d_hidden, n_heads=n_heads)
    temp_decoder = DecoderLayer(d_model=d_model, d_hidden=d_hidden, n_heads=n_heads)

    encoder_output, _ = temp_encoder(encoder_input=encoder_input, mask=encoder_mask)
    print(encoder_output.shape)
    decoder_output, *_ = temp_decoder(encoder_output=encoder_output, decoder_input=decoder_output,
                                      self_attn_mask=self_attn_mask, enc_attn_mask=dec_attn_mask)
    print(decoder_output.shape)


