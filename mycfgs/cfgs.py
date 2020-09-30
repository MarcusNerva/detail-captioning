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
    parser.add_argument('--data_dir', type=str, default='/home/hanhuaye/PythonProject/detail-captioning/mydata')
    parser.add_argument('--videos_dir', type=str, default='/home/hanhuaye/PythonProject/detail-captioning/mydata/videos')
    parser.add_argument('--frames_dir', type=str, default='/home/hanhuaye/PythonProject/detail-captioning/mydata/picked_frames')

    parser.add_argument('--raw_res2d_dir', type=str, default='/home/hanhuaye/PythonProject/detail-captioning/mydata/res2d_features')
    parser.add_argument('--res2d_dir', type=str, default='/home/hanhuaye/PythonProject/detail-captioning/mydata/res2d_features/features')
    parser.add_argument('--i3d_dir', type=str, default='/home/hanhuaye/PythonProject/detail-captioning/mydata/i3d_features')
    parser.add_argument('--relation_dir', type=str, default='/home/hanhuaye/PythonProject/detail-captioning/mydata/relation_features')
    parser.add_argument('--object_dir', type=str, default='/home/hanhuaye/PythonProject/detail-captioning/mydata/object_features')
    parser.add_argument('--res2d_mask_path', type=str, default='/home/hanhuaye/PythonProject/detail-captioning/mydata/res2d_mask.npy')
    parser.add_argument('--i3d_mask_path', type=str, default='/home/hanhuaye/PythonProject/detail-captioning/mydata/i3d_mask.npy')
    parser.add_argument('--seq_mask_path', type=str, default='/home/hanhuaye/PythonProject/detail-captioning/mydata/seq_mask.pkl')
    parser.add_argument('--json_path', type=str, default='/home/hanhuaye/PythonProject/detail-captioning/mydata/videodatainfo_2017.json')
    parser.add_argument('--torchtext_path', type=str, default='/home/hanhuaye/PythonProject/detail-captioning/mydata/torchtext.pkl')
    parser.add_argument('--seq_dict_path', type=str, default='/home/hanhuaye/PythonProject/detail-captioning/mydata/seq_dict.pkl')
    parser.add_argument('--numberic_dict_path', type=str, default='/home/hanhuaye/PythonProject/detail-captioning/mydata/numberic_dict.pkl')
    parser.add_argument('--checkpoints_dir', type=str, default='/home/hanhuaye/PythonProject/detail-captioning/checkpoints')

    parser.add_argument('--train_datastore_dir', type=str, default='/home/hanhuaye/PythonProject/detail-captioning/mydata/train')
    parser.add_argument('--valid_datastore_dir', type=str, default='/home/hanhuaye/PythonProject/detail-captioning/mydata/valid')
    parser.add_argument('--test_datastore_dir', type=str, default='/home/hanhuaye/PythonProject/detail-captioning/mydata/test')


    """
    =====================DPP Settings=====================
    """
    parser.add_argument('--n_pick', type=int, default=20)
    # parser.add_argument('--eps', type=float, default=0.01)
    parser.add_argument('--i3d_eps', type=float, default=0)
    parser.add_argument('--frames_eps', type=float, default=0)

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
    =====================Transformer Settings=====================
    """
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--d_hidden', type=int, default=2048)
    parser.add_argument('--d_features', type=int, default=512 * 5)
    parser.add_argument('--trans_dropout', type=float, default=0.3)


    """
    =====================Model Settings=====================
    """
    parser.add_argument('--seq_length', type=int, default=30)
    parser.add_argument('--vocab_size', type=int, default=-1)
    parser.add_argument('--beam_size', type=int, default=1)
    parser.add_argument('--sample_max', type=int, default=1)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--bos', type=str, default='<bos>')
    parser.add_argument('--eos', type=str, default='<eos>')
    parser.add_argument('--pad', type=str, default='<eos>')
    parser.add_argument('--unk', type=str, default='<unk>')


    """
    =====================Training Settings=====================
    """
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--grad_clip', type=float, default=0.1)
    parser.add_argument('--save_checkpoint_every', type=int, default=500)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--continue_to_train', action='store_true')

    parser.add_argument('--learning_rate', type=float, default=3e-4, help='learning rate')
    parser.add_argument('--learning_rate_decay_start', type=int, default=6, help='after how many iteration begin learning rate decay')
    parser.add_argument('--learning_rate_decay_every', type=int, default=4, help='for every x iteration learning rate have to decay')
    parser.add_argument('--learning_rate_decay_rate', type=float, default=0.5)
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')

    parser.add_argument('--self_critical_after', type=int, default=9, help='after train x epochs use self_critical strategy')
    
    parser.add_argument('--visualize_every', type=int, default=10, help='show us loss every x iteration')

    args = parser.parse_args()
    return args


