time=$(date "+%Y-%m-%d_%H:%M:%S")

python train.py \
-m outputs/v1_anchor/horse_blender/backpack_test \
-s /home/jiahao/GS-IR/datasets/Gshader/horse_blender \
--iterations 35000 \
--gamma \
--eval  \
--pbr_training