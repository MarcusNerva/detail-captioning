
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

echo "If don't PickUp, please set 0.
If PickUp fully, please set 1.
If PickUp with threshold, please set 2.
Please choose the manual of PickUp:  " 
manual=0

if [ $manual -eq 0 ];then
    datastore_dir="/home/hanhuaye/PythonProject/detail-captioning/datastore/msvd/not_pickup"
elif [ $manual -eq 1 ];then
    datastore_dir="/home/hanhuaye/PythonProject/detail-captioning/datastore/msvd/pickup_full"
elif [ $manual -eq 2 ];then
    datastore_dir="/home/hanhuaye/PythonProject/detail-captioning/datastore/msvd/pickup_threshold"
    echo "Please set frame_eps and i3d_eps respectively
    (#####WARNING#####: To ensure the consistency of the input eps and eps in cfgs.py, please check cfgs.py before inputing eps): "
    frame_eps=0
    i3d_eps=0
    sub_dir="res2d_"${frame_eps}"_i3d_"${i3d_eps}
    datastore_dir=${datastore_dir}"/"${sub_dir}
else
    echo "############WRONG INPUT(this shell scripts is going to end now)############"
    exit 1
fi


conda activate py37
echo "########################data preparing begins, and now time is {$(date "+%Y-%m-%d %H:%M:%S")}########################"

if [ ! -d ./msvd_log ];then
    echo "msvd_log not exists. I am going to make dir"
    mkdir msvd_log
fi
python /home/hanhuaye/PythonProject/detail-captioning/myscripts/data_processing_scripts/extract_msvd_frames.py \
    > msvd_log/extract_frames.log || exit 1
echo "===============extracting frames is finished! Now time is {$(date "+%Y-%m-%d %H:%M:%S")}==============="

python /home/hanhuaye/PythonProject/detail-captioning/myscripts/data_processing_scripts/extract_scene_feats.py \
    --frames_dir /home/hanhuaye/PythonProject/detail-captioning/mydata/msvd_data/picked_frames \
    --raw_res2d_dir /home/hanhuaye/PythonProject/detail-captioning/mydata/msvd_data/res2d_features \
    > msvd_log/extract_scene_feats.log || exit 1
echo "===============extracting scene feats is finished! Now time is {$(date "+%Y-%m-%d %H:%M:%S")}==============="

conda activate py27

MSVD_features_dir="/home/hanhuaye/PythonProject/opensource/MSDN/MSVD_features/"
MSVD_boxes_dir="/home/hanhuaye/PythonProject/opensource/MSDN/MSVD_boxes"
if [ -d ${MSVD_features_dir} ];then
    echo "####### Clear up MSVD_features_dir #######"
    rm -rf ${MSVD_features_dir}"/"
    mkdir -p ${MSVD_features_dir}
fi

if [ -d ${MSVD_boxes_dir} ];then
    echo "####### Clear up MSVD_boxes_dir #######"
    rm -rf ${MSVD_boxes_dir}"/"
    mkdir -p ${MSVD_boxes_dir}
fi

python /home/hanhuaye/PythonProject/opensource/MSDN/train_hdn.py \
 --resume_training --disable_language_model --rnn_type LSTM_normal \
 --dataset_option=normal  --MPS_iter=1 \
 --evaluate \
 --total_images_dir /home/hanhuaye/PythonProject/detail-captioning/mydata/msvd_data/picked_frames \
 --total_feature_dir /home/hanhuaye/PythonProject/opensource/MSDN/MSVD_features \
 --total_box_dir /home/hanhuaye/PythonProject/opensource/MSDN/MSVD_boxes \
 || exit 1

echo "===============extracting MSDN feats is finished! Now time is {$(date "+%Y-%m-%d %H:%M:%S")}==============="

conda activate py37

python /home/hanhuaye/PythonProject/opensource/MSDN/merge_msvd_features.py || exit 1
echo "===============merging feats is finished! Now time is {$(date "+%Y-%m-%d %H:%M:%S")}==============="

python /home/hanhuaye/PythonProject/opensource/MSDN/merge_msvd_boxes.py || exit 1
echo "===============merging boxes is finished! Now time is {$(date "+%Y-%m-%d %H:%M:%S")}==============="

python /home/hanhuaye/PythonProject/detail-captioning/myscripts/data_processing_scripts/merge_msvd_scene_feats.py || exit 1
echo "===============merging res2d_feats is finished! Now time is {$(date "+%Y-%m-%d %H:%M:%S")}==============="

python /home/hanhuaye/PythonProject/detail-captioning/myscripts/data_processing_scripts/rearrange_object.py \
    --now_msvd \
    --msdn_features /home/hanhuaye/PythonProject/opensource/MSDN/MSVD_features/features \
    --box_boundings /home/hanhuaye/PythonProject/opensource/MSDN/MSVD_boxes/boxes \
    || exit 1
echo "===============rearranging objects & relation feats is finished! Now time is {$(date "+%Y-%m-%d %H:%M:%S")}==============="

i3d_msvd_feats_store=/home/hanhuaye/PythonProject/opensource/kinetics-i3d/i3d_msvd_feats
i3d_msvd_feats_dir=/home/hanhuaye/PythonProject/opensource/kinetics-i3d/data/i3d_msvd_feats

if [ ! -d ${i3d_msvd_feats_store} ] || [ ! -d ${i3d_msvd_feats_dir} ]; then
    echo "i am going to extract i3d feats for msvd dataset!"
    mkdir -p ${i3d_msvd_feats_dir}
    python /home/hanhuaye/PythonProject/opensource/kinetics-i3d/extract_msvd_i3d.py || exit 1
    cp -r ${i3d_msvd_feats_dir} ${i3d_msvd_feats_store}
else
    cp -r ${i3d_msvd_feats_store} ${i3d_msvd_feats_dir}
fi

python /home/hanhuaye/PythonProject/detail-captioning/myscripts/data_processing_scripts/pick_up_msvd_i3d_feats.py || exit 1
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

echo "Greetings, Henry! I am glad to inform you that data of MSVD Dataset extraction is finished, and now is copying data to data_store!" \
    | mail -s "MSVD Data Extraction Finished!" hadrianus_1@163.com
