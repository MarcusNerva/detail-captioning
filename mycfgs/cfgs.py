#!/usr/bin/env python
# coding=utf-8

def get_total_settings():
    import argparse
    parser = argparse.ArgumentParser()


    """
    =====================General Settings=====================
    """
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--drop_prob', type=float, default=0.5)


    """
    =====================Data & Model Path Settings=====================
    """
    parser.add_argument('--data_dir', type=str, default='/disks/lilaoshi666/hanhua.ye/detail-captioning/mydata')
    parser.add_argument('--videos_dir', type=str, default='/disks/lilaoshi666/hanhua.ye/detail-captioning/mydata/videos')
    parser.add_argument('--frames_dir', type=str, default='/disks/lilaoshi666/hanhua.ye/detail-captioning/mydata/picked_frames')

    parser.add_argument('--res2d_dir', type=str, default='/disks/lilaoshi666/hanhua.ye/detail-captioning/mydata/res2d_features')
    parser.add_argument('--i3d_dir', type=str, default='/disks/lilaoshi666/hanhua.ye/detail-captioning/mydata/i3d_features')
    parser.add_argument('--relation_dir', type=str, default='/disks/lilaoshi666/hanhua.ye/detail-captioning/mydata/relation_features')
    parser.add_argument('--object_dir', type=str, default='/disks/lilaoshi666/hanhua.ye/detail-captioning/mydata/object_features')
    parser.add_argument('--res2d_mask_path', type=str, default='/disks/lilaoshi666/hanhua.ye/detail-captioning/mydata/res2d_mask.npy')
    parser.add_argument('--i3d_mask_path', type=str, default='/disks/lilaoshi666/hanhua.ye/detail-captioning/mydata/i3d_mask.npy')
    parser.add_argument('--seq_mask_path', type=str, default='/disks/lilaoshi666/hanhua.ye/detail-captioning/mydata/seq_mask.npy')
    parser.add_argument('--json_path', type=str, default='/disks/lilaoshi666/hanhua.ye/detail-captioning/mydata/videodatainfo_2017.json')
    parser.add_argument('--torchtext_path', type=str, default='/disks/lilaoshi666/hanhua.ye/detail-captioning/mydata/torchtext.pkl')
    parser.add_argument('--seq_dict_path', type=str, default='/disks/lilaoshi666/hanhua.ye/detail-captioning/mydata/seq_dict.pkl')
    parser.add_argument('--numberic_dict_path', type=str, default='/disks/lilaoshi666/hanhua.ye/detail-captioning/mydata/numberic_dict.npy')
    parser.add_argument('--checkpoints_dir', type=str, default='/disks/lilaoshi666/hanhua.ye/detail-captioning/checkpoints')


    """
    =====================DPP Settings=====================
    """
    parser.add_argument('--n_pick', type=int, default=20)
    parser.add_argument('--eps', type=float, default=0.01)


    """
    =====================Encoder Settings=====================
    """
    parser.add_argument('--n_objs', type=int, default=5)
    parser.add_argument('--res2d_size', type=int, default=2048)
    parser.add_argument('--i3d_size', type=int, default=1024)
    parser.add_argument('--relation_size', type=int, default=1024)
    parser.add_argument('--object_size', type=int, default=2048)
    parser.add_argument('--rnn_size', type=int, default=512)
    parser.add_argument('--length', type=int, default=20)


    """
    =====================Decoder Settings=====================
    """
    parser.add_argument('--word_size', type=int, default=512)



    """
    =====================Model Settings=====================
    """
    parser.add_argument('--seq_length', type=int, default=30)
    parser.add_argument('--vocab_size', type=int, default=-1)
    parser.add_argument('--beam_size', type=int, default=1)
    parser.add_argument('--sample_max', type=int, default=1)
    parser.add_argument('--bos', type=str, default='<bos>')
    parser.add_argument('--eos', type=str, default='<eos>')
    parser.add_argument('--pad', type=str, default='<eos>')


    """
    =====================Training Settings=====================
    """
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--grad_clip', type=float, default=0.1)
    parser.add_argument('--save_checkpoint_every', type=int, default=500)
    parser.add_argument('--patience', type=int, default=30)

    parser.add_argument('--learning_rate', type=float, default=3e-4, help='learning rate')
    parser.add_argument('--learning_rate_decay_start', type=int, default=-1, help='after how many iteration begin learning rate decay')
    parser.add_argument('--learning_rate_decay_every', type=int, default=4, help='for every x iteration learning rate have to decay')
    parser.add_argument('--learning_rate_decay_rate', type=float, default=0.5)
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')

    parser.add_argument('--self_critical_after', type=int, default=-1, help='after train x epochs use self_critical strategy')
    
    parser.add_argument('--visualize_every', type=int, default=10, help='show us loss every x iteration')

    args = parser.parse_args()
    return args


