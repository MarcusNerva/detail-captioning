#!/usr/bin/env python
# coding=utf-8
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchnet import meter
from collections import OrderedDict
import os
import numpy as np

from coco_caption.pycocoevalcap.cider.cider import Cider
from ..mymodels import CaptionModel, DatasetMSRVTT, collate_fn
from loss import LanguageModelCriterion, RewardCriterion
from ..mycfgs.cfgs import get_total_settings
from ..mytools import Visualizer
from eval import eval, decode_idx

def get_self_critical_reward(args, model, res2ds, i3ds, relations, objects, res_mask, i3d_mask, probability_sample, ground_truths):
    batch_size = args.batch_size
    seq_length = args.seq_length
    double_batch_size = batch_size * 2
    
    greedy_sample, _ = model.sample(res2ds, i3ds, relations, objects, res_mask, i3d_mask)
    res, gts = OrderedDict(), OrderedDict()
    greedy_sample, probability_sample = greedy_sample.cpu().numpy(), probability_sample.cpu().numpy()
    
    for i in range(batch_size):
        res[i] = [decode_idx(probability_sample[i])]
    for i in range(batch_size, double_batch_size):
        res[i] = [decode_idx(greedy_sample[i - batch_size])]

    for i in range(batch_size):
        gts[i] = [decode_idx(single_gts) for single_gts in ground_truths[i]]
    gts = {i: gts[i % batch_size] for i in range(double_batch_size)}

    assert len(gts.keys()) == len(res.keys()), 'len of gts.keys is not equal to that of res.keys'
    avg_cider_score, cider_score = Cider().compute_score(gts=gts, res=res)
    cider_score = np.array(cider_score)
    reward = cider_score[:batch_size] - cider_score[batch_size:]
    reward = np.repeat(reward[:, np.newaxis], seq_length, axis=1)
    return reward

def clip_gradient(optimizer, grad_clip):
    for group in optimizer:
        for param in group['params']:
            if param.grad is None: continue
            param.grad.data.clamp_(-grad_clip, grad_clip)

def set_learning_rate(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr

def train(args):
    batch_size = args.batch_size
    seed = args.seed
    checkpoint_dir = args.checkpoint_dir
    grad_clip = args.grad_clip
    learning_rate = args.learning_rate
    learning_rate_decay_start = args.learning_rate_decay_start
    learning_rate_decay_every = args.learning_rate_decay_every
    learning_rate_decay_rate = args.learning_rate_decay_rate
    weight_decay = args.weight_decay
    patience = args.patience
    save_checkpoint_every = args.save_checkpoint_every
    visualize_every = args.visualize_every
    self_critical_after = args.self_critical_after

    best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    dataset = DatasetMSRVTT(mode='train', args=args)
    valid_dataset = DatasetMSRVTT(mode='valid', args=args)
    args.pad_idx = dataset.get_pad_idx()
    args.bos_idx = dataset.get_bos_idx()
    args.eos_idx = dataset.get_eos_idx()
    args.n_vocab = dataset.get_n_vocab()

    dataloader = DataLoader(dataset, batch_size, shuffle=True, collate_fn=collate_fn)
    vis = Visualizer(env='train model')
    model = CaptionModel(args)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_meter = meter.AverageValueMeter()
    crit = LanguageModelCriterion()
    rl_crit = RewardCriterion()
    
    patience_cnt = 0
    epoch = 0
    sc_signal = False
    best_score = None

    model.to(device)
    model.train()

    while True:
        if patience_cnt >= patience: break
        loss_meter.reset()
        model.train()
        
        if learning_rate_decay_start != -1 and epoch > learning_rate_decay_every:
            frac = int((epoch - learning_rate_decay_start) // learning_rate_decay_every)
            decay_factor = learning_rate_decay_rate ** frac
            current_lr = learning_rate * decay_factor
            set_learning_rate(optimizer, current_lr)
        
        if self_critical_after != -1 and epoch >= self_critical_after:
            sc_signal = True
            save_checkpoint_every = 250

        for i, (res2d, i3d, relation, object_, res2d_mask, i3d_mask, numberic, mask, seq) in enumerate(dataloader):

            res2d = res2d.to(device)
            i3d = i3d.to(device)
            relation = relation.to(device)
            object_ = object_.to(device)
            res2d_mask = res2d_mask.to(device)
            i3d_mask = i3d_mask.to(device)
            numberic = numberic.to(device)
            mask = mask.to(device)

            optimizer.zero_grad()
            if not sc_signal:
                preds = model(res2d, i3d, relation, object_, numberic, res2d_mask, i3d_mask, mask)
                loss = crit(preds, numberic, mask, args.eos_idx)
            else:
                probability_sample, sample_logprobs = model.sample(res2d, i3d, relation, object_, res2d_mask, i3d_mask, False)
                reward = get_self_critical_reward(model, res2d, i3d, relation, object_, res2d_mask, i3d_mask, probability_sample, seq)
                reward = torch.from_numpy(reward).float()
                reward = reward.to(device)
                loss = rl_crit(sample_logprobs, probability_sample, reward)
            loss.backward()
            clip_gradient(optimizer, grad_clip)
            optimizer.step()
            train_loss = loss.detach()
            loss_meter.add(train_loss.item())

            if i % visualize_every == 0:
                vis.plot('train_loss', loss_meter.value()[0])
                information = 'best_score is' + (str(best_score) if best_score is not None else '0.0')
                information += ('reward is' if sc_flag else 'loss is ') + str(train_loss.item())
                vis.log(information)

            is_best = False
            if (i + 1) % save_checkpoint_every == 0:
                language_state = eval(args, model, valid_dataset, device)
                current_score = language_state['CIDEr']
                vis.log('{}'.format('cider score is ' + str(current_score)) + ' ' + '+ ' + str(i))

                if best_score is None or best_score < current_score:
                    is_best = True
                    best_score = current_score
                    patience_cnt = 0

                if is_best:
                    torch.save(model.state_dict(), best_model_path)

        epoch += 1


if __name__ == '__main__':
    args = get_total_settings()
    train(args)
