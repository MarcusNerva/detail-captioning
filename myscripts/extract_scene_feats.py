#!/usr/bin/env python
# coding=utf-8
import numpy as np
import torch
from torchvision.models.resnet import model_urls, load_state_dict_from_url, ResNet, Bottleneck
from torchvision import transforms
from PIL import Image
import os
import glob

mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
normalize = transforms.Normalize(mean=mean, std=std)
trans = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
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
        state_dict = load_state_dict_from_url(models_url[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model

def get_resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)

def extract_features(args, model, device):
    image_path = args.image_path
    output_path = args.output_path

    model.to(device)
    model.eval()
    with torch.no_grad():
        images_path_list = glob.glob(os.path.join(image_path, '*.jpg'))
        for item in images_path_list:
            img = Image.open(item)
            img = trans(img).unsqueeze(0).to(device)
            feat = model(img)
            feat = feat.squeeze()

            img_name = item.split('/')[-1].split('.')[0]
            save_path = os.path.join(output_path, img_name + '.npy')
            np.save(save_path, feat)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default='/disks/lilaoshi666/hanhua.ye/MSDN/MSRVTT_image')
    parser.add_argument('--output_path', type=str, default='/disks/lilaoshi666/hanhua.ye/detail-captioning/data/scene_features')
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    resnext101_32x8d = get_resnext101_32x8d(pretrained=True)
    extract_features(args, resnext101_32x8d, device)
