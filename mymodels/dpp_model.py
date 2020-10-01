#!/usr/bin/env python
# coding=utf-8
import os
import glob
import numpy as np
import torch
from torchvision import transforms
from PIL import Image


class DPPModel:
    def __init__(self, mode, feature_dir, settings, device, extract_model=None):
        super(DPPModel, self).__init__()
        if mode not in ['frame', 'i3d']:
            raise NotImplementedError
        if mode == 'frame' and extract_model is None:
            raise Exception('You forget to pass in extract_model')
        if mode == 'i3d' and extract_model is not None:
            raise Exception('No need to pass in extract_model to pick i3d')

        self.n_items = -1 #Does not know yet
        self.kernel_matrix = None #which is going to be built in build_kernel_matrix()
        self.n_pick = settings['n_pick']
        self.i3d_eps = settings['i3d_eps']
        self.frames_eps = settings['frames_eps']
        self.mode = mode
        self.feature_dir = feature_dir
        self.extract_model = extract_model
        self.device = device

        self.build_kernel_matrix()
        self.eps = self.frames_eps if self.mode == 'frame' else self.i3d_eps

    def build_kernel_matrix(self):
        if self.mode == 'frame':
            if self.frames_eps < 0.0:
                self.image_path_list = glob.glob(os.path.join(self.feature_dir, '*.jpg'))
                self.n_items = len(self.image_path_list)
                return

            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            normalize = transforms.Normalize(mean=mean, std=std)
            trans = transforms.Compose([
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                normalize
            ])
            model = self.extract_model
            model.eval()
            model.to(self.device)
            image_path_list = glob.glob(os.path.join(self.feature_dir, '*.jpg'))
            image_path_list = sorted(image_path_list)
            feats_store = []
            
            with torch.no_grad():
                for item in image_path_list:
                    img = Image.open(item)
                    img = trans(img).unsqueeze(0).to(self.device)
                    feat = model(img)
                    feat = feat.cpu().numpy()
                    feats_store.append(feat)
            feats_embed = np.concatenate(feats_store, axis=0)
            self.kernel_matrix = feats_embed @ feats_embed.T
            self.n_items = feats_embed.shape[0]

        else:
            if self.i3d_eps < 0.0:
                feats_embed = np.load(self.feature_dir)
                self.n_items = feats_embed.shape[0]
                return

            feats_embed = np.load(self.feature_dir)
            self.kernel_matrix = feats_embed @ feats_embed.T
            self.n_items = feats_embed.shape[0]

    def dpp(self):
        if self.eps < 0.0:
            Yg = np.linspace(0, self.n_items, num=self.n_pick, endpoint=False, dtype=np.int).tolist()
            return Yg

        c = np.zeros((self.n_pick, self.n_items))
        d = np.copy(np.diag(self.kernel_matrix))
        j = np.argmax(d)
        Yg = []
        Yg.append(j)
        Z = list(range(self.n_items))
        
        it = 0
        while len(Yg) < self.n_pick:
            Z_Y = set(Z).difference(set(Yg))
            for i in Z_Y:
                if it == 0:
                    ei = self.kernel_matrix[j, i] / np.sqrt(d[j])
                else:
                    ei = (self.kernel_matrix[j, i] - np.dot(c[:it, j], c[:it, i])) / np.sqrt(d[j])
                c[it, i] = ei
                d[i] = d[i] - ei * ei
            d[j] = 0
            j = np.argmax(d)
            if self.eps > 0. and d[j] < self.eps:
                break
            Yg.append(j)
            it += 1
        return Yg

