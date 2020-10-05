#!/usr/bin/env python
# coding=utf-8
import torch
import torch.nn as nn
from .decoder import Decoder, Decoder_Part
from .encoder import Encoder, Encoder_Part

class CaptionModel(nn.Module):
    def __init__(self, args):
        super(CaptionModel, self).__init__()
        
        self.args = args
        self.length = args.length
        self.seq_length = args.seq_length
        self.rnn_size = args.rnn_size
        self.eos_idx = args.eos_idx
        self.bos_idx = args.bos_idx
        self.unk_idx = args.unk_idx
        
        seed = args.seed
        vocab_size = args.n_vocab
        word_size = args.word_size
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        self.embed_h = nn.Linear(2 * self.rnn_size, self.rnn_size)
        self.embed_c = nn.Linear(2 * self.rnn_size, self.rnn_size)
        self.encoder = Encoder(args)
        self.decoder = Decoder(args)
        self.embed = nn.Embedding(vocab_size, word_size)
        self.logit = nn.Linear(word_size, vocab_size)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.vocab_size = vocab_size
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embed.weight.data.uniform_(-initrange, initrange)
        self.logit.bias.data.fill_(0)
        self.logit.weight.data.uniform_(-initrange, initrange)

    def init_state(self, res2d, i3d, res_mask, i3d_mask):
        _res2d = torch.sum(res2d.cpu(), dim=1)
        _mask = torch.sum(res_mask.cpu().double(), dim=1, keepdims=True)
        _res2d = _res2d / _mask

        _i3d = torch.sum(i3d.cpu(), dim=1)
        _mask = torch.sum(i3d_mask.cpu().double(), dim=1, keepdims=True)
        _i3d = _i3d / _mask
        
        _feats = torch.cat([_res2d, _i3d], dim=-1).unsqueeze(0).to(self.device).double()
        state_h = self.embed_h(_feats)
        state_c = self.embed_c(_feats)

        return (state_h, state_c)

    def forward(self, res2d, i3d, relation, objects, word_seq, res_mask, i3d_mask, word_mask):
        res2d, i3d, relation, objects = self.encoder(res2d, i3d, relation, objects, res_mask, i3d_mask)
        state = self.init_state(res2d, i3d, res_mask, i3d_mask)
        outputs = []

        for i in range(word_seq.shape[1]):
            if i > 0 and word_seq[:, i].sum() == 0:
                output_word = torch.zeros(word_seq.shape[0], self.vocab_size).to(self.device)
                outputs.append(output_word)
                continue

            it = word_seq[:, i].clone()
            xt = self.embed(it)
            xt_mask = word_mask[:, i].unsqueeze(1)
            output, state = self.decoder(res2d, i3d, relation, objects, xt, state, res_mask, i3d_mask, xt_mask)
            output_word = torch.log_softmax(self.logit(output), dim=1)
            outputs.append(output_word)

        ret_seq = torch.stack(outputs, dim=1)
        return ret_seq

    def beam_step(self, t, logprobs, beam_seq, beam_seq_logprobs, beam_logprobs_sum, state):
        args = self.args
        beam_size = args.beam_size

        probs, idx = torch.sort(logprobs, dim=1, descending=True)
        candidates = []
        rows = beam_size if t >= 1 else 1
        cols = min(beam_size, probs.size(1))

        for r in range(rows):
            for c in range(cols):
                tmp_logprob = probs[r, c]
                tmp_sum = beam_logprobs_sum[r] + tmp_logprob
                tmp_idx = idx[r, c]
                candidates.append({'sum': tmp_sum, 'logprob': tmp_logprob, 'ix': tmp_idx, 'beam': r})

        candidates = sorted(candidates, key=lambda x: -x['sum'])
        prev_seq = beam_seq[:, :t].clone()
        prev_seq_probs = beam_seq_logprobs[:, :t].clone()
        prev_logprobs_sum = beam_logprobs_sum.clone()
        new_state = [_.clone() for _ in state]

        for i in range(beam_size):
            candidate_i = candidates[i]
            beam = candidate_i['beam']
            ix = candidate_i['ix']
            logprob = candidate_i['logprob']

            beam_seq[i, :t] = prev_seq[beam, :]
            beam_seq_logprobs[i, :t] = prev_seq_probs[beam, :]
            beam_seq[i, t] = ix
            beam_seq_logprobs[i, t] = logprob
            beam_logprobs_sum[i] = prev_logprobs_sum[beam] + logprob
            for j in range(len(new_state)):
                new_state[j][:, i, :] = state[j][:, beam, :]

        return beam_seq, beam_seq_logprobs, beam_logprobs_sum, new_state

    def beam_search(self, res2d, i3d, relation, objects, res_mask, i3d_mask, state):
        args = self.args
        beam_size = args.beam_size

        beam_seq = torch.LongTensor(beam_size, self.seq_length).fill_(self.eos_idx)
        beam_seq_logprobs = torch.DoubleTensor(beam_size, self.seq_length).zero_()
        beam_logprobs_sum = torch.zeros(beam_size)
        ret = []

        it = torch.LongTensor(beam_size).fill_(self.bos_idx).to(self.device)
        xt = self.embed(it).to(self.device)
        xt_mask = torch.ones([beam_size, 1]).double().to(self.device)
        output, state = self.decoder(res2d, i3d, relation, objects, xt, state, res_mask, i3d_mask, xt_mask)
        logprob = torch.log_softmax(self.logit(output), dim=1)

        for t in range(self.seq_length):
            # suppress UNK tokens in the decoding. So the probs of 'UNK' are extremely low
            logprob[:, self.unk_idx] = logprob[:, self.unk_idx] - 1000.0
            beam_seq, beam_seq_logprobs, beam_logprobs_sum, state = self.beam_step(t=t,
                                                                                   logprobs=logprob,
                                                                                   beam_seq=beam_seq,
                                                                                   beam_seq_logprobs=beam_seq_logprobs,
                                                                                   beam_logprobs_sum=beam_logprobs_sum,
                                                                                   state=state)

            for j in range(beam_size):
                if beam_seq[j, t] == self.eos_idx or t == self.seq_length - 1:
                    final_beam = {
                        'seq': beam_seq[j, :].clone(),
                        'seq_logprob': beam_seq_logprobs[j, :].clone(),
                        'sum_logprob': beam_logprobs_sum[j].clone()
                    }
                    ret.append(final_beam)
                    beam_logprobs_sum[j] = -1000.0

            it = beam_seq[:, t].to(self.device)
            xt = self.embed(it).to(self.device)
            xt_mask = torch.ones([beam_size, 1]).double().to(self.device)
            output, state = self.decoder(res2d, i3d, relation, objects, xt, state, res_mask, i3d_mask, xt_mask)
            logprob = torch.log_softmax(self.logit(output), dim=1)

        ret = sorted(ret, key=lambda x: -x['sum_logprob'])[:beam_size]
        return ret

    def sample_beam(self, res2ds, i3ds, relations, objects, res_mask, i3d_mask):
        args = self.args
        beam_size = args.beam_size
        batch_size = res2ds.size(0)

        seq = torch.LongTensor(batch_size, self.seq_length).fill_(self.eos_idx)
        seq_probabilities = torch.DoubleTensor(batch_size, self.seq_length)
        done_beam = [[] for _ in range(batch_size)]

        for i in range(batch_size):
            single_res2d = res2ds[i] #(20, 512)
            single_i3d = i3ds[i] #(20, 512)
            single_relation = relations[i] #(20, 10, 512)
            single_object = objects[i] #(20, 10, 1024)
            single_res_mask = res_mask[i] #(20)
            single_i3d_mask = i3d_mask[i] #(20)

            single_res2d = single_res2d.expand(beam_size, single_res2d.size(0), single_res2d.size(1))
            single_i3d = single_i3d.expand(beam_size, single_i3d.size(0), single_i3d.size(1))
            single_relation = single_relation.expand(beam_size, single_relation.size(0), single_relation.size(1), single_relation.size(2))
            single_object = single_object.expand(beam_size, single_object.size(0), single_object.size(1), single_object.size(2))
            single_res_mask = single_res_mask.expand(beam_size, single_res_mask.size(0))
            single_i3d_mask = single_i3d_mask.expand(beam_size, single_i3d_mask.size(0))
            state = self.init_state(single_res2d, single_i3d, single_res_mask, single_i3d_mask)

            done_beam[i] = self.beam_search(single_res2d, single_i3d, single_relation, single_object, 
                                            single_res_mask, single_i3d_mask, state)
            seq[i] = done_beam[i][0]['seq']
            seq_probabilities[i] = done_beam[i][0]['seq_logprob']

        return seq, seq_probabilities

    def sample(self, res2ds, i3ds, relations, objects, res_mask, i3d_mask, is_sample_max=True):
        args = self.args
        sample_max = args.sample_max if is_sample_max else 0
        beam_size = args.beam_size
        batch_size = res2ds.shape[0]
        temperature = args.temperature

        res2ds, i3ds, relations, objects = self.encoder(res2ds, i3ds, relations, objects, res_mask, i3d_mask)

        if beam_size > 1:
            return self.sample_beam(res2ds, i3ds, relations, objects, res_mask, i3d_mask)

        state = self.init_state(res2ds, i3ds, res_mask, i3d_mask)
        seq, seq_probabilities = [], []
        
        for t in range(self.seq_length):
            if t == 0:
                it = res2ds.new(batch_size).long().fill_(self.bos_idx)
            elif sample_max:
                sampleLogprobs, it = torch.max(log_probabilities.detach(), 1)
                it = it.view(-1).long()
            else:
                prev_probabilities = torch.exp(torch.div(log_probabilities.detach(), temperature))
                it = torch.multinomial(prev_probabilities, 1)
                sampleLogprobs = log_probabilities.gather(1, it)
                it = it.view(-1).long()

            xt = self.embed(it)

            if t >= 1:
                if t == 1:
                    unfinished = it > 0
                else:
                    unfinished = unfinished * (it > 0)
                # if unfinished.sum() == 0: break
                it = it * unfinished.type_as(it)
                seq.append(it)
                seq_probabilities.append(sampleLogprobs.view(-1))
                
            if t == 0:
                xt_mask = torch.ones([batch_size, 1]).double()
            else:
                xt_mask = unfinished.unsqueeze(-1).double()

            xt = xt.to(self.device)
            xt_mask = xt_mask.to(self.device)
            output, state = self.decoder(res2ds, i3ds, relations, objects, xt, state, res_mask, i3d_mask, xt_mask)
            log_probabilities = torch.log_softmax(self.logit(output.double()), dim=1)

        seq.append(it.new(batch_size).long().fill_(self.eos_idx))
        seq_probabilities.append(sampleLogprobs.view(-1))
        return torch.cat([_.unsqueeze(1) for _ in seq], 1), torch.cat([_.unsqueeze(1) for _ in seq_probabilities], 1)

class CaptionModel_Part(nn.Module):
    def __init__(self, args):
        super(CaptionModel_Part, self).__init__()

        self.args = args
        self.length = args.length
        self.seq_length = args.seq_length
        self.rnn_size = args.rnn_size
        self.eos_idx = args.eos_idx
        self.bos_idx = args.bos_idx
        self.unk_idx = args.unk_idx

        seed = args.seed
        vocab_size = args.n_vocab
        word_size = args.word_size
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        self.embed_h = nn.Linear(2 * self.rnn_size, self.rnn_size)
        self.embed_c = nn.Linear(2 * self.rnn_size, self.rnn_size)
        self.encoder = Encoder_Part(args)
        self.decoder = Decoder_Part(args)
        self.embed = nn.Embedding(vocab_size, word_size)
        self.logit = nn.Linear(word_size, vocab_size)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.vocab_size = vocab_size
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embed.weight.data.uniform_(-initrange, initrange)
        self.logit.bias.data.fill_(0)
        self.logit.weight.data.uniform_(-initrange, initrange)

    def init_state(self, res2d, i3d, res_mask, i3d_mask):
        _res2d = torch.sum(res2d.cpu(), dim=1)
        _mask = torch.sum(res_mask.cpu().double(), dim=1, keepdims=True)
        _res2d = _res2d / _mask

        _i3d = torch.sum(i3d.cpu(), dim=1)
        _mask = torch.sum(i3d_mask.cpu().double(), dim=1, keepdims=True)
        _i3d = _i3d / _mask

        _feats = torch.cat([_res2d, _i3d], dim=-1).unsqueeze(0).to(self.device).double()
        state_h = self.embed_h(_feats)
        state_c = self.embed_c(_feats)

        return (state_h, state_c)

    def forward(self, res2d, i3d, relation, objects, word_seq, res_mask, i3d_mask, word_mask):
        res2d, i3d = self.encoder(res2d, i3d, res_mask, i3d_mask)
        state = self.init_state(res2d, i3d, res_mask, i3d_mask)
        outputs = []

        for i in range(word_seq.shape[1]):
            if i > 0 and word_seq[:, i].sum() == 0:
                output_word = torch.zeros(word_seq.shape[0], self.vocab_size).to(self.device)
                outputs.append(output_word)
                continue

            it = word_seq[:, i].clone()
            xt = self.embed(it)
            xt_mask = word_mask[:, i].unsqueeze(1)
            output, state = self.decoder(res2d, i3d, xt, state, res_mask, i3d_mask, xt_mask)
            output_word = torch.log_softmax(self.logit(output), dim=1)
            outputs.append(output_word)

        ret_seq = torch.stack(outputs, dim=1)
        return ret_seq

    def beam_step(self, t, logprobs, beam_seq, beam_seq_logprobs, beam_logprobs_sum, state):
        args = self.args
        beam_size = args.beam_size

        probs, idx = torch.sort(logprobs, dim=1, descending=True)
        candidates = []
        rows = beam_size if t >= 1 else 1
        cols = min(beam_size, probs.size(1))

        for r in range(rows):
            for c in range(cols):
                tmp_logprob = probs[r, c]
                tmp_sum = beam_logprobs_sum[r] + tmp_logprob
                tmp_idx = idx[r, c]
                candidates.append({'sum': tmp_sum, 'logprob': tmp_logprob, 'ix': tmp_idx, 'beam': r})

        candidates = sorted(candidates, key=lambda x: -x['sum'])
        prev_seq = beam_seq[:, :t].clone()
        prev_seq_probs = beam_seq_logprobs[:, :t].clone()
        prev_logprobs_sum = beam_logprobs_sum.clone()
        new_state = [_.clone() for _ in state]

        for i in range(beam_size):
            candidate_i = candidates[i]
            beam = candidate_i['beam']
            ix = candidate_i['ix']
            logprob = candidate_i['logprob']

            beam_seq[i, :t] = prev_seq[beam, :]
            beam_seq_logprobs[i, :t] = prev_seq_probs[beam, :]
            beam_seq[i, t] = ix
            beam_seq_logprobs[i, t] = logprob
            beam_logprobs_sum[i] = prev_logprobs_sum[beam] + logprob
            for j in range(len(new_state)):
                new_state[j][:, i, :] = state[j][:, beam, :]

        return beam_seq, beam_seq_logprobs, beam_logprobs_sum, new_state

    def beam_search(self, res2d, i3d, relation, objects, res_mask, i3d_mask, state):
        args = self.args
        beam_size = args.beam_size

        beam_seq = torch.LongTensor(beam_size, self.seq_length).fill_(self.eos_idx)
        beam_seq_logprobs = torch.DoubleTensor(beam_size, self.seq_length).zero_()
        beam_logprobs_sum = torch.zeros(beam_size)
        ret = []

        it = torch.LongTensor(beam_size).fill_(self.bos_idx).to(self.device)
        xt = self.embed(it).to(self.device)
        xt_mask = torch.ones([beam_size, 1]).double().to(self.device)
        output, state = self.decoder(res2d, i3d, xt, state, res_mask, i3d_mask, xt_mask)
        logprob = torch.log_softmax(self.logit(output), dim=1)

        for t in range(self.seq_length):
            # suppress UNK tokens in the decoding. So the probs of 'UNK' are extremely low
            logprob[:, self.unk_idx] = logprob[:, self.unk_idx] - 1000.0
            beam_seq, beam_seq_logprobs, beam_logprobs_sum, state = self.beam_step(t=t,
                                                                                   logprobs=logprob,
                                                                                   beam_seq=beam_seq,
                                                                                   beam_seq_logprobs=beam_seq_logprobs,
                                                                                   beam_logprobs_sum=beam_logprobs_sum,
                                                                                   state=state)

            for j in range(beam_size):
                if beam_seq[j, t] == self.eos_idx or t == self.seq_length - 1:
                    final_beam = {
                        'seq': beam_seq[j, :].clone(),
                        'seq_logprob': beam_seq_logprobs[j, :].clone(),
                        'sum_logprob': beam_logprobs_sum[j].clone()
                    }
                    ret.append(final_beam)
                    beam_logprobs_sum[j] = -1000.0

            it = beam_seq[:, t].to(self.device)
            xt = self.embed(it).to(self.device)
            xt_mask = torch.ones([beam_size, 1]).double().to(self.device)
            output, state = self.decoder(res2d, i3d, xt, state, res_mask, i3d_mask, xt_mask)
            logprob = torch.log_softmax(self.logit(output), dim=1)

        ret = sorted(ret, key=lambda x: -x['sum_logprob'])[:beam_size]
        return ret

    def sample_beam(self, res2ds, i3ds, relations, objects, res_mask, i3d_mask):
        args = self.args
        beam_size = args.beam_size
        batch_size = res2ds.size(0)

        seq = torch.LongTensor(batch_size, self.seq_length).fill_(self.eos_idx)
        seq_probabilities = torch.DoubleTensor(batch_size, self.seq_length)
        done_beam = [[] for _ in range(batch_size)]

        for i in range(batch_size):
            single_res2d = res2ds[i]
            single_i3d = i3ds[i]
            single_res_mask = res_mask[i]
            single_i3d_mask = i3d_mask[i]

            single_res2d = single_res2d.expand(beam_size, single_res2d.size(0), single_res2d.size(1))
            single_i3d = single_i3d.expand(beam_size, single_i3d.size(0), single_i3d.size(1))
            single_relation = single_relation.expand(beam_size, single_relation.size(0), single_relation.size(1), single_relation.size(2))
            single_object = single_object.expand(beam_size, single_object.size(0), single_object.size(1), single_object.size(2))
            single_res_mask = single_res_mask.expand(beam_size, single_res_mask.size(0))
            single_i3d_mask = single_i3d_mask.expand(beam_size, single_i3d_mask.size(0))
            state = self.init_state(single_res2d, single_i3d, single_res_mask, single_i3d_mask)

            done_beam[i] = self.beam_search(single_res2d, single_i3d, single_relation, single_object,
                                            single_res_mask, single_i3d_mask, state)
            seq[i] = done_beam[i][0]['seq']
            seq_probabilities[i] = done_beam[i][0]['seq_logprob']

        return seq, seq_probabilities

    def sample(self, res2ds, i3ds, relations, objects, res_mask, i3d_mask, is_sample_max=True):
        args = self.args
        sample_max = args.sample_max if is_sample_max else 0
        beam_size = args.beam_size
        batch_size = res2ds.shape[0]
        temperature = args.temperature

        res2ds, i3ds = self.encoder(res2ds, i3ds, res_mask, i3d_mask)

        if beam_size > 1:
            return self.sample_beam(res2ds, i3ds, relations, objects, res_mask, i3d_mask)

        state = self.init_state(res2ds, i3ds, res_mask, i3d_mask)
        seq, seq_probabilities = [], []

        for t in range(self.seq_length):
            if t == 0:
                it = res2ds.new(batch_size).long().fill_(self.bos_idx)
            elif sample_max:
                sampleLogprobs, it = torch.max(log_probabilities.detach(), 1)
                it = it.view(-1).long()
            else:
                prev_probabilities = torch.exp(torch.div(log_probabilities.detach(), temperature))
                it = torch.multinomial(prev_probabilities, 1)
                sampleLogprobs = log_probabilities.gather(1, it)
                it = it.view(-1).long()

            xt = self.embed(it)

            if t >= 1:
                if t == 1:
                    unfinished = it > 0
                else:
                    unfinished = unfinished * (it > 0)
                # if unfinished.sum() == 0: break
                it = it * unfinished.type_as(it)
                seq.append(it)
                seq_probabilities.append(sampleLogprobs.view(-1))

            if t == 0:
                xt_mask = torch.ones([batch_size, 1]).double()
            else:
                xt_mask = unfinished.unsqueeze(-1).double()

            xt = xt.to(self.device)
            xt_mask = xt_mask.to(self.device)
            output, state = self.decoder(res2ds, i3ds, xt, state, res_mask, i3d_mask, xt_mask)
            log_probabilities = torch.log_softmax(self.logit(output.double()), dim=1)

        seq.append(it.new(batch_size).long().fill_(self.eos_idx))
        seq_probabilities.append(sampleLogprobs.view(-1))
        return torch.cat([_.unsqueeze(1) for _ in seq], 1), torch.cat([_.unsqueeze(1) for _ in seq_probabilities], 1)
