#!/bin/bash

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/hanhuaye/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/hanhuaye/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/home/hanhuaye/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/hanhuaye/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

conda activate py37

CUDA_VISIBLE_DEVICES=0 python /home/hanhuaye/PythonProject/detail-captioning/myscripts/data_processing_scripts/extract_frames.py > log/extract_frames.log
exit 1
echo "===============extracting frames is finished!==============="

CUDA_VISIBLE_DEVICES=0 python /home/hanhuaye/PythonProject/detail-captioning/myscripts/data_processing_scripts/extract_scene_feats.py > log/extract_scene_feats.log 
exit 1
echo "===============extracting scene feats is finished!==============="


conda activate py27
CUDA_VISIBLE_DEVICES=0 /home/hanhuaye/PythonProject/opensource/MSDN/eval.sh 
exit 1
echo "===============extracting MSDN feats is finished==============="
conda activate py37

sh /home/hanhuaye/PythonProject/opensource/MSDN/merge.sh 
exit 1
echo "===============merging feats is finished==============="

sh /home/hanhuaye/PythonProject/opensource/MSDN/merge_boxes.sh
exit 1
echo "===============merging boxes is finished==============="

python /home/hanhuaye/PythonProject/detail-captioning/myscripts/data_processing_scripts/rearrange_object.py > log/rearrange_object.log
exit 1
echo "===============rearranging objects & relation feats is finished==============="

python /home/hanhuaye/PythonProject/detail-captioning/myscripts/data_processing_scripts/pick_up_i3d_feats.py > log/pick_up_i3d_feats.log 
exit 1
echo "===============picking up i3d feats is finished==============="
