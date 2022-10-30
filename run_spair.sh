# python CATs/train_semimatch.py --batch-size 8 --lr 0.000001 --gpu_id 0 \
# --freeze False --pretrained 'CATs/pretrained/spair_no_freeze/epoch_11.pth' --name_exp 'mutaul_nearest_dynamic_keyout_gt_mask_spair_from_11' --keyout 0.2 \
# --contrastive_gt_mask --refined_corr_filtering 'mutual' --additional_weak \
# --benchmark spair --hyperpixel '[0,8,20,21,26,28,29,30]' --dynamic_unsup

python CATs/train_semimatch.py --batch-size 8 --lr 0.000001 --gpu_id 2 \
--freeze False --pretrained 'CATs/pretrained/spair_no_freeze/epoch_11.pth' --name_exp 'dynamic_spair_from_11_reimpl' --keyout 0.2 \
--contrastive_gt_mask --refined_corr_filtering 'mutual' --additional_weak \
--benchmark spair --hyperpixel '[0,8,20,21,26,28,29,30]' --dynamic_unsup

