import os
import shutil
import subprocess
import glob
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torchvision.models.mobilenet import model_urls, MobileNetV2, load_state_dict_from_url
import sys
sys.path.append('../../')
sys.path.append('../../../')
from mymodels import DPPModel
from mycfgs.cfgs import get_total_settings

class MyMobileNet(MobileNetV2):
    def __init__(self,
                num_classes=1000,
                 width_mult=1.0,
                 inverted_residual_setting=None,
                 round_nearest=8,
                 block=None,
                 norm_layer=None):
        super(MyMobileNet, self).__init__()

    def _forward_impl(self, x):
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        return x


def mobilenet_v2(pretrained=False, progress=True, **kwargs):
    model = MyMobileNet()
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def extract_frames(video_path, dst):
    with open(os.devnull, 'w') as ffmpeg_log:
        if os.path.exists(dst):
            print('clean up ' + dst)
            shutil.rmtree(dst)
        os.makedirs(dst)
        video_to_frames_command = ['ffmpeg', '-y', '-i', video_path, '-r', '25', '-qscale:v', '2',
                                   '{0}/%06d.jpg'.format(dst)]
        subprocess.call(video_to_frames_command, stdout=ffmpeg_log, stderr=ffmpeg_log)

if __name__ == '__main__':
    args = get_total_settings()
    dst = os.path.join(args.frames_dir, 'cache')
    videos_dir = args.videos_dir
    frames_dir = args.frames_dir
    if os.path.exists(frames_dir):
        shutil.rmtree(frames_dir)
    os.makedirs(frames_dir)

    settings = vars(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _mobilenet = mobilenet_v2(pretrained=True)
    mode = 'frame'

    video_list = glob.glob(os.path.join(videos_dir, '*.mp4'))
    video_list = sorted(video_list, key=lambda video_path: int(video_path.split('/')[-1].split('.')[0][5:]))
    res2d_mask = np.zeros((len(video_list), args.length), dtype=bool)

    for i, (video_path) in enumerate(video_list):
        extract_frames(video_path, dst)
        frames_list = glob.glob(os.path.join(dst, '*.jpg'))
        frames_list = sorted(frames_list)
        dpp = DPPModel(mode, dst, settings, device, _mobilenet)
        Yg = dpp.dpp()
        Yg = sorted(Yg)
        res2d_mask[i, :len(Yg)] = True
        print(Yg)

        for idx in Yg:
            jpg_path = frames_list[idx]
            jpg_name = jpg_path.split('/')[-1]
            new_jpg_path = os.path.join(frames_dir, '%06d' % i + '_' + jpg_name)
            print(jpg_path)
            print(new_jpg_path)
            shutil.copyfile(jpg_path, new_jpg_path)

    np.save(args.res2d_mask_path, res2d_mask)
