#!/bin/bash

# >>> conda init >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$(CONDA_REPORT_ERRORS=false '/home/hanhua.ye/anaconda3/bin/conda' shell.bash hook 2> /dev/null)"
if [ $? -eq 0 ]; then
    \eval "$__conda_setup"
else
    if [ -f "/home/hanhua.ye/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/home/hanhua.ye/anaconda3/etc/profile.d/conda.sh"
        CONDA_CHANGEPS1=false conda activate base
    else
        \export PATH="/home/hanhua.ye/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda init <<<

conda activate py37

CUDA_VISIBLE_DEVICES=0 python /disks/lilaoshi666/hanhua.ye/detail-captioning/myscripts/data_processing_scripts/extract_frames.py > log/extract_frames.log
exit 1
echo "===============extracting frames is finished!==============="

CUDA_VISIBLE_DEVICES=0 python /disks/lilaoshi666/hanhua.ye/detail-captioning/myscripts/data_processing_scripts/extract_scene_feats.py > log/extract_scene_feats.log 
exit 1
echo "===============extracting scene feats is finished!==============="


conda activate python27
CUDA_VISIBLE_DEVICES=0 /disks/lilaoshi666/hanhua.ye/MSDN/eval.sh 
exit 1
echo "===============extracting MSDN feats is finished==============="
conda activate py37

sh /disks/lilaoshi666/hanhua.ye/MSDN/merge.sh 
exit 1
echo "===============merging feats is finished==============="

sh /disks/lilaoshi666/hanhua.ye/MSDN/merge_boxes.sh
exit 1
echo "===============merging boxes is finished==============="

python /disks/lilaoshi666/hanhua.ye/detail-captioning/myscripts/data_processing_scripts/rearrange_object.py > log/rearrange_object.log
exit 1
echo "===============rearranging objects & relation feats is finished==============="

python /disks/lilaoshi666/hanhua.ye/detail-captioning/myscripts/data_processing_scripts/pick_up_i3d_feats.py > log/pick_up_i3d_feats.log 
exit 1
echo "===============picking up i3d feats is finished==============="
