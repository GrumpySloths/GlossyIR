time=$(date "+%Y-%m-%d_%H:%M:%S")
detect_anomaly=false 
voxel_size=0.001
update_init_factor=16
appearance_dim=0
ratio=1

anomaly_flag=""
if [ "$detect_anomaly" = true ]; then
    echo "Detecting anomaly"
    anomaly_flag="--detect_anomaly"
fi

echo "anomaly_flag: $anomaly_flag"

python train.py \
-m outputs/v1_anchor/horse_blender/${time} \
-s /home/jiahao/GS-IR/datasets/Gshader/horse_blender/ \
--iterations 30000 \
--eval $anomaly_flag \
--voxel_size ${voxel_size} \
--update_init_factor ${update_init_factor} \
--appearance_dim ${appearance_dim} \
--ratio ${ratio} &&
python train.py \
-m outputs/v1_anchor/horse_blender/${time} \
-s /home/jiahao/GS-IR/datasets/Gshader/horse_blender \
--iterations 35000 \
--gamma \
--eval \
--pbr_training $anomaly_flag