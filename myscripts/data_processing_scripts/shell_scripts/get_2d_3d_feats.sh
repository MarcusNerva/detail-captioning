
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

nohup cp -r /home/hanhuaye/PythonProject/opensource/kinetics-i3d/i3d_feats \
    /home/hanhuaye/PythonProject/opensource/kinetics-i3d/data/i3d_feats > log/cp_i3d.log 2>&1 &

read -p "If don't PickUp, please enter 0.
If PickUp fully, please enter 1.
If PickUp with threshold, please enter 2.
Please choose the manual of PickUp:  " manual

if [ $manual -eq 0 ];then
    datastore_dir="/home/hanhuaye/PythonProject/detail-captioning/datastore/not_pickup"
elif [ $manual -eq 1 ];then
    datastore_dir="/home/hanhuaye/PythonProject/detail-captioning/datastore/pickup_full"
elif [ $manual -eq 2 ];then
    datastore_dir="/home/hanhuaye/PythonProject/detail-captioning/datastore/pickup_threshold"
    read -p "Please enter frame_eps and i3d_eps respectively
    (#####WARNING#####: To ensure the consistency of the input eps and eps in cfgs.py, please check cfgs.py before inputing eps): " frame_eps i3d_eps
    sub_dir="res2d_"${frame_eps}"_i3d_"${i3d_eps}
    datastore_dir=${datastore_dir}"/"${sub_dir}
else
    echo "############WRONG INPUT(this shell scripts is going to end now)############"
    exit 1
fi


conda activate py37
echo "########################data preparing begins, and now time is {$(date "+%Y-%m-%d %H:%M:%S")}########################"

CUDA_VISIBLE_DEVICES=0 python /home/hanhuaye/PythonProject/detail-captioning/myscripts/data_processing_scripts/extract_frames.py > log/extract_frames.log || exit 1
echo "===============extracting frames is finished! Now time is {$(date "+%Y-%m-%d %H:%M:%S")}==============="

python /home/hanhuaye/PythonProject/detail-captioning/myscripts/data_processing_scripts/extract_scene_feats.py || exit 1
echo "===============extracting scene feats is finished! Now time is {$(date "+%Y-%m-%d %H:%M:%S")}==============="

conda activate py27
python /home/hanhuaye/PythonProject/opensource/MSDN/train_hdn.py \
 --resume_training --disable_language_model --rnn_type LSTM_normal \
 --dataset_option=normal  --MPS_iter=1 \
 --evaluate \
 || exit 1

echo "===============extracting MSDN feats is finished! Now time is {$(date "+%Y-%m-%d %H:%M:%S")}==============="

conda activate py37

python /home/hanhuaye/PythonProject/opensource/MSDN/merge_features.py || exit 1
echo "===============merging feats is finished! Now time is {$(date "+%Y-%m-%d %H:%M:%S")}==============="

python /home/hanhuaye/PythonProject/opensource/MSDN/merge_boxes.py || exit 1
echo "===============merging boxes is finished! Now time is {$(date "+%Y-%m-%d %H:%M:%S")}==============="

python /home/hanhuaye/PythonProject/detail-captioning/myscripts/data_processing_scripts/merge_scene_feats.py || exit 1
echo "===============merging res2d_feats is finished! Now time is {$(date "+%Y-%m-%d %H:%M:%S")}==============="

python /home/hanhuaye/PythonProject/detail-captioning/myscripts/data_processing_scripts/rearrange_object.py || exit 1
echo "===============rearranging objects & relation feats is finished! Now time is {$(date "+%Y-%m-%d %H:%M:%S")}==============="

python /home/hanhuaye/PythonProject/detail-captioning/myscripts/data_processing_scripts/pick_up_i3d_feats.py || exit 1
echo "===============picking up i3d feats is finished! Now time is {$(date "+%Y-%m-%d %H:%M:%S")}==============="


echo "########################data transforming begins, and now time is {$(date "+%Y-%m-%d %H:%M:%S")}########################"
if [ ! -d ${datastore_dir} ];then
    echo "${datastore_dir} not exists. I am going to make dir!"
    mkdir -p ${datastore_dir}
fi

datastore_object_features=${datastore_dir}"/object_features"
datastore_relation_features=${datastore_dir}"/relation_features"
datastore_res2d_features=${datastore_dir}"/res2d_features"
datastore_i3d_features=${datastore_dir}"/i3d_features"
datastore_res2d_mask=${datastore_dir}"/res2d_mask.npy"
datastore_i3d_mask=${datastore_dir}"/i3d_mask.npy"

data_dir="/home/hanhuaye/PythonProject/detail-captioning/mydata"
origin_object_features=${data_dir}"/object_features"
origin_relation_features=${data_dir}"/relation_features"
origin_res2d_features=${data_dir}"/res2d_features/features"
origin_i3d_features="/home/hanhuaye/PythonProject/opensource/kinetics-i3d/data/i3d_feats"
origin_res2d_mask=${data_dir}"/res2d_mask.npy"
origin_i3d_mask=${data_dir}"/i3d_mask.npy"

nohup cp -r ${origin_object_features} ${datastore_object_features} > log/cp_object_features.log 2>&1 &
nohup cp -r ${origin_relation_features} ${datastore_relation_features} > log/cp_relation_features.log 2>&1 &
nohup cp -r ${origin_res2d_features} ${datastore_res2d_features} > log/cp_res2d_features.log 2>&1 &
nohup cp -r ${origin_i3d_features} ${datastore_i3d_features} > log/cp_i3d_features.log 2>&1 &
nohup cp ${origin_res2d_mask} ${datastore_res2d_mask} > log/cp_res2d_mask.log 2>&1 &
nohup cp ${origin_i3d_mask} ${datastore_i3d_mask} > log/cp_i3d_mask.log 2>&1 &


