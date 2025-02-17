time=$(date "+%Y-%m-%d_%H:%M:%S")
voxel_size=0.001
update_init_factor=16
appearance_dim=0
ratio=1

python train.py \
-m outputs/v1_anchor/horse_blender/${time} \
-s /home/jiahao/GS-IR/datasets/Gshader/horse_blender/ \
--iterations 30000 \
--eval \
--voxel_size ${voxel_size} \
--update_init_factor ${update_init_factor} \
--appearance_dim ${appearance_dim} \
--ratio ${ratio} 