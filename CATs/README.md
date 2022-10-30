# CVPR_semi_matching

 python train.py --gpu_id 0 --p_cutoff 0.9 --batch-size 4 --epochs 200 --semi_softmax_corr_temp 0.1 --hyperpixel [2,17,21,22,25,26,28] --warmup_epoch 10 \
 --pretrained 'snapshots/2021_10_28_10_55/epoch_10_warmUp.pth' --loss_mode 'contrastive' --name_exp 'auto' --interpolation_mode 'nearest' \
 --seed 2 --refined_corr_filtering 'mutual' --use_fbcheck_mask True
