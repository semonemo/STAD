#!/bin/bash
# ver0-nrgbd / ver1-per_frame / ver2-SA_pred / ver3-SA_NO_pred / ver4-aggr / ver5-aggr_occ / ver6-aggr_kr / ver7-aggr_r /

### STAD
python3 train_KVNet.py \
		--exp_name ver4-aggr_silog \
		--nepoch 30 \
		--RNet \
		--sigma_soft_max 10\
		--LR 1e-4 \
		--t_win 2 \
		--d_min 1 \
		--d_max 60 \
		--feature_dim 64 \
		--ndepth 64 \
		--grad_clip \
		--grad_clip_max 2. \
		--dataset kitti \
		--dataset_path /data/kitti \
		--loss_type silog \
		--batch_size 16

#### STAD-frame
#python3 train_KVNet.py \
#		--exp_name ver1-per_frame_silog \
#		--nepoch 30 \
#		--RNet \
#		--sigma_soft_max 10\
#		--LR 1e-4 \
#		--t_win 0 \
#		--d_min 1 \
#		--d_max 60 \
#		--feature_dim 64 \
#		--ndepth 64 \
#		--grad_clip \
#		--grad_clip_max 2. \
#		--dataset kitti \
#		--dataset_path /data/kitti \
#		--loss_type silog \
#		--batch_size 16
#
#### nrgbd-silog
#python3 train_KVNet.py \
#		--exp_name ver0-nrgbd-silog  \
#		--nepoch 30 \
#		--RNet \
#		--sigma_soft_max 10\
#		--LR 1e-4 \
#		--t_win 2 \
#		--d_min 1 \
#		--d_max 60 \
#		--feature_dim 64 \
#		--ndepth 64 \
#		--grad_clip \
#		--grad_clip_max 2. \
#		--dataset kitti \
#		--dataset_path /data/kitti \
#		--loss_type silog \
#		--batch_size 16
#
#### nrgbd
#python3 train_KVNet.py \
#		--exp_name ver0-nrgbd  \
#		--nepoch 30 \
#		--RNet \
#		--sigma_soft_max 10\
#		--LR 1e-4 \
#		--t_win 2 \
#		--d_min 1 \
#		--d_max 60 \
#		--feature_dim 64 \
#		--ndepth 64 \
#		--grad_clip \
#		--grad_clip_max 2. \
#		--dataset kitti \
#		--dataset_path /data/kitti \
#		--loss_type NLL \
#		--batch_size 16
#
