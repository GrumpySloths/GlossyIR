python train.py \
-m outputs/lego/ \
-s datasets/TensoIR/lego/ \
--start_checkpoint outputs/lego/chkpnt30000.pth \
--iterations 35000 \
--eval \
--gamma \
--indirect