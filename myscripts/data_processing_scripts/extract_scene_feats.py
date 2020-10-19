#!/usr/bin/env python
# coding=utf-8
import numpy as np
import torch
import shutil
from torchvision.models.resnet import model_urls, load_state_dict_from_url, ResNet, Bottleneck
from torchvision import transforms
from PIL import Image
import os
import glob
import sys
sys.path.append('../../')
sys.path.append('../../../')
from mycfgs.cfgs import get_total_settings

mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
normalize = transforms.Normalize(mean=mean, std=std)
trans = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    normalize
])

class MyResNet(ResNet):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False, groups=1, width_per_group=64, 
                 replace_stride_with_dilation=None, norm_layer=None):
        super(MyResNet, self).__init__(block, layers, num_classes, zero_init_residual, groups, width_per_group, 
                                       replace_stride_with_dilation, norm_layer)
    
    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x

def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = MyResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model

def get_resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)

def extract_features(args, model, device):
    image_dir = args.frames_dir
    output_dir = args.raw_res2d_dir
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    model.eval()
    model.to(device)
    # with torch.no_grad():
    images_path_list = glob.glob(os.path.join(image_dir, '*.jpg'))
    with torch.no_grad():
        for item in images_path_list:
            img = Image.open(item)
            img = trans(img).unsqueeze(0).to(device)
            feat = model(img)
            feat = feat.squeeze().cpu().numpy()

            img_name = item.split('/')[-1].split('.')[0]
            save_path = os.path.join(output_dir, img_name + '.npy')
            np.save(save_path, feat)

if __name__ == '__main__':
    args = get_total_settings()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device is ', device)

    resnext101_32x8d = get_resnext101_32x8d(pretrained=True)
    extract_features(args, resnext101_32x8d, device)
    print('=======================finish==========================')
