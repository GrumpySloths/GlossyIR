python baking.py \
-m ./outputs/Gshader/horse_blender \
--checkpoint outputs/Gshader/horse_blender/chkpnt30000.pth \
--bound 1.5 \
--occlu_res 128 \
--occlusion 0.25 &&
python train.py \
-m ./outputs/Gshader/horse_blender/ \
-s datasets/Gshader/horse_blender/ \
--start_checkpoint outputs/Gshader/horse_blender/chkpnt30000.pth \
--iterations 35000 \
--eval \
--gamma \
--indirect