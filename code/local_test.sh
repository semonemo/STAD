#!/bin/bash
# ver0-nrgbd / ver1-per_frame / ver2-SA_pred / ver3-SA_NO_pred / ver4-aggr / ver5-aggr_occ / ver6-aggr_kr / ver7-aggr_r /
CUDA_VISIBLE_DEVICES=2 python3 test_KVNet.py \
    --t_win 2 \
 	--d_min 1 \
 	--d_max 60 \
    --ndepth 64 \
    --sigma_soft_max 10\
    --feature_dim 64 \
 	--dataset kitti \
 	--dataset_path /data/kitti \
    --split_file ./mdataloader/kitti_split/test_eigen.txt \
    --model_path /data/out/kitti/ver4-aggr_silog.tar




## test monodepth2
## works on conda 'neuralrgbd' (torch 0.4.0)
# CUDA_VISIBLE_DEVICES=2 python3 SOTAs/monodepth2/test_monodepth2.py \
#    --exp_name te/ \
#    --t_win 2 \
#    --d_min 1 \
#    --d_max 60 \
#    --ndepth 64 \
#    --sigma_soft_max 10\
#    --feature_dim 64 \
#    --dataset kitti \
#    --dataset_path /data/kitti \
#    --split_file ./mdataloader/kitti_split/test_eigen.txt \

## test BTS
## need to turn on conda 'bts' (torch 1.2.0)
# CUDA_VISIBLE_DEVICES=1 python3 SOTAs/BTS/test_bts.py \
#    --exp_name te/ \
#    --t_win 2 \
#    --d_min 1 \
#    --d_max 60 \
#    --ndepth 64 \
#    --sigma_soft_max 10\
#    --feature_dim 64 \
#    --dataset kitti \
#    --dataset_path /data/kitti \
#    --split_file ./mdataloader/kitti_split/test_eigen.txt \
