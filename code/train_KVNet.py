'''
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license

author: Chao Liu <chaoliu1@cs.cmu.edu>
'''

# train both D-net and KV-net and R-net #
import torch

torch.backends.cudnn.benchmark = True

import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

import mdataloader.misc as dl_misc
import mdataloader.batch_loader as batch_loader
import mdataloader.scanNet as dl_scanNet

import mutils.misc as m_misc
import warping.homography as warp_homo
import numpy as np
import math

import models.basic as m_basic
from models.KVNET import KVNET as orig_kvnet
import models.KVNET_transformer as m_kvnet

import utils.models as utils_model
import torch.optim as optim
from tensorboardX import SummaryWriter

import time

import train_utils.train_KVNet as train_KVNet

import os, sys
import train_utils.Logger as Logger
import wandb
from datetime import datetime
import random

seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)
random.seed(seed)

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


def get_current_lr(optimizer):
    for param_group in optimizer.param_groups:
        current_lr = param_group['lr']
    return current_lr


def add_noise2pose(src_cam_poses_in, noise_level=.2):
    '''
    noise_level - gaussian_sigma / norm_r r, gaussian_sigma/ norm_t for t
    add Gaussian noise to the poses:
    for R: add in the unit-quaternion space
    for t: add in the raw space
    '''

    src_cam_poses_out = torch.zeros(src_cam_poses_in.shape)
    src_cam_poses_out[:, :, 3, 3] = 1.
    # for each batch #
    for ibatch in range(src_cam_poses_in.shape[0]):
        src_cam_poses_perbatch = src_cam_poses_in[ibatch, ...]
        for icam in range(src_cam_poses_perbatch.shape[0]):
            src_cam_pose = src_cam_poses_perbatch[icam, ...]

            # convert to unit quaternion #
            r = m_misc.Rotation2UnitQ(src_cam_pose[:3, :3].cuda())
            t = src_cam_pose[:3, 3]

            # add noise to r and t #
            sigma_r = noise_level * r.norm()
            sigma_t = noise_level * t.norm()
            r = r + torch.randn(r.shape).cuda() * sigma_r
            t = t + torch.randn(t.shape) * sigma_t

            # put back in to src_cam_poses_out #
            src_cam_poses_out[ibatch, icam, :3, :3] = m_misc.UnitQ2Rotation(r).cpu()
            src_cam_poses_out[ibatch, icam, :3, 3] = t

    return src_cam_poses_out


def check_datArray_pose(dat_array):
    '''
    Check data array pose/dmap for scan-net.
    If invalid pose then use the previous pose.
    Input: data-array: will be modified via reference.
    Output:
    False: not valid, True: valid
    '''
    if_valid = True
    for dat in dat_array:
        if isinstance(dat['dmap'], int):
            if_valid = False
            break

        elif np.isnan(dat['extM'].min()) or np.isnan(dat['extM'].max()):
            if_valid = False
            break

    return if_valid


def main():
    import argparse
    print('Parsing the arguments...')
    parser = argparse.ArgumentParser()

    # exp name #
    parser.add_argument('--exp_name', required=True, type=str,
                        help='The name of the experiment. Used to naming the folders')

    # nepoch #
    parser.add_argument('--nepoch', required=True, type=int, help='# of epochs to run')

    # if pretrain #
    parser.add_argument('--pre_trained', action='store_true', default=False,
                        help='If use the pre-trained model; (False)')

    # logging #
    parser.add_argument('--TB_add_img_interv', type=int, default=50,
                        help='The inerval for log one training image')

    parser.add_argument('--pre_trained_model_path', type=str,
                        default='.', help='The pre-trained model path for\
                        KV-net')

    # model saving #
    parser.add_argument('--save_model_interv', type=int, default=500,
                        help='The interval of iters to save the model; default: 5000')

    # tensorboard #
    parser.add_argument('--TB_fldr', type=str, default='runs',
                        help='The tensorboard logging root folder; default: runs')

    # about training #
    parser.add_argument('--RNet', action='store_true', help='if use refinement net to improve the depth resolution',
                        default=True)
    parser.add_argument('--weight_var', default=.001, type=float,
                        help='weight for the variance loss, if we use L1 loss')
    parser.add_argument('--pose_noise_level', default=0, type=float,
                        help='Noise level for pose. Used for training with pose noise')
    parser.add_argument('--frame_interv', default=5, type=int, help='frame interval')
    parser.add_argument('--LR', default=.001, type=float, help='Learning rate')
    parser.add_argument('--t_win', type=int, default=2, help='The radius of the temporal window; default=2')
    parser.add_argument('--d_min', type=float, default=0, help='The minimal depth value; default=0')
    parser.add_argument('--d_max', type=float, default=15, help='The maximal depth value; default=15')
    parser.add_argument('--ndepth', type=int, default=128, help='The # of candidate depth values; default= 128')
    parser.add_argument('--grad_clip', action='store_true', help='if clip the gradient')
    parser.add_argument('--grad_clip_max', type=float, default=2, help='the maximal norm of the gradient')
    parser.add_argument('--sigma_soft_max', type=float, default=500., help='sigma_soft_max, default = 500.')
    parser.add_argument('--feature_dim', type=int, default=64,
                        help='The feature dimension for the feature extractor; default=64')
    parser.add_argument('--batch_size', type=int, default=0,
                        help='The batch size for training; default=0, means batch_size=nGPU')
    parser.add_argument('--loss_type', type=str, default='NLL', help='loss type: {NLL, L1, silog,}')

    # about dataset #
    parser.add_argument('--dataset', type=str, default='scanNet', help='Dataset name: {scanNet, kitti,}')
    parser.add_argument('--dataset_path', type=str, default='.', help='Path to the dataset')
    parser.add_argument('--change_aspect_ratio', action='store_true', default=False,
                        help='If we want to change the aspect ratio. This option is only useful for KITTI')

    # para config. #
    args = parser.parse_args()
    exp_name = args.exp_name
    saved_model_path = f'/data/out/{args.dataset}/{args.exp_name}/saved_models/'

    if args.batch_size == 0:
        batch_size = torch.cuda.device_count()
    else:
        batch_size = args.batch_size

    d_candi = np.linspace(args.d_min, args.d_max, args.ndepth)
    d_candi_dmap_ref = d_candi

    # saving model config.#
    m_misc.m_makedir(saved_model_path)


    # Logging
    now = datetime.now().strftime('%m-%d-%H-%M')
    run_name = f'{exp_name}-{now}'
    wandb.init(project="stad", name=f'Train/{run_name}', config=args)

    # Initialize data-loader, model and optimizer #

    # ===== Dataset selection ======== #
    dataset_path = args.dataset_path
    if args.dataset == 'kitti':
        import mdataloader.kitti as dl_kitti
        dataset_init = dl_kitti.KITTI_dataset

        if not dataset_path == '.':
            fun_get_paths = lambda traj_indx: dl_kitti.get_paths(traj_indx,
                                                                 split_txt='./mdataloader/kitti_split/train_eigen.txt',
                                                                 mode='train',
                                                                 database_path_base=dataset_path)
        else:  # use default database path
            fun_get_paths = lambda traj_indx: dl_kitti.get_paths(traj_indx,
                                                                 split_txt='./mdataloader/kitti_split/train_eigen.txt',
                                                                 mode='train')

        # img_size = [1248, 380]
        if not args.change_aspect_ratio:  # we will keep the aspect ratio and do cropping
            img_size = [768, 256]
            crop_w = 384
            # crop_w = None

        else:  # we will change the aspect ratio and NOT do cropping
            img_size = [384, 256]
            # img_size = [512, 256]
            # img_size = [624, 256]
            crop_w = None

        n_scenes, _, _, _, _ = fun_get_paths(0)
        traj_Indx = np.arange(0, n_scenes)

    else:
        raise Exception('dataset not implemented ')
    fldr_path, img_paths, dmap_paths, poses, intrin_path = fun_get_paths(0)
    if args.dataset == 'kitti':
        dataset = dataset_init(True, img_paths, dmap_paths, poses,
                               intrin_path=intrin_path, img_size=img_size, digitize=True,
                               d_candi=d_candi_dmap_ref, resize_dmap=.25, crop_w=crop_w)
    else:
        dataset = dataset_init(True, img_paths, dmap_paths, poses,
                               intrin_path=intrin_path, img_size=img_size, digitize=True,
                               d_candi=d_candi_dmap_ref, resize_dmap=.25)
    # ================================ #

    print('Initnializing the KV-Net')

    version_num = int(args.exp_name.split('-')[0].split('ver')[-1])
    if version_num == 0:
        print("**train with original nrgbd(V0) method")
        model_version = orig_kvnet
        train_KVNet_train = train_KVNet.train
        no_pred_DPV = False


    elif version_num == 1:
        print("**train with per-frame(V1) method")
        model_version = m_kvnet.KVNET1
        train_KVNet_train = train_KVNet.train_per_frame
        no_pred_DPV = True

    elif version_num == 2:
        print("**train with SA with pred DPV (V2) method")
        model_version = m_kvnet.KVNET2
        train_KVNet_train = train_KVNet.train
        no_pred_DPV = False

    elif version_num == 3:
        print("**test with SA without pred DPV (V3) method")
        model_version = m_kvnet.KVNET3
        train_KVNet_train = train_KVNet.train_no_pred_DPV
        no_pred_DPV = True

    elif version_num == 4:
        print("**train with aggregation(V4) method")
        model_version = m_kvnet.KVNET4
        train_KVNet_train = train_KVNet.train_no_pred_DPV
        no_pred_DPV = True

    elif version_num == 5:
        print("**train with aggregation + occ mask (V5) method")
        model_version = m_kvnet.KVNET5
        train_KVNet_train = train_KVNet.train_no_pred_DPV
        no_pred_DPV = True

    elif version_num == 6:
        print("**train with aggregate+KR(V6) method")
        model_version = m_kvnet.KVNET6
        train_KVNet_train = train_KVNet.train_no_pred_DPV
        no_pred_DPV = True

    elif version_num == 7:
        print("**train with aggregate+R(V7) method")
        model_version = m_kvnet.KVNET7
        train_KVNet_train = train_KVNet.train_no_pred_DPV
        no_pred_DPV = True


    else:
        raise NotImplementedError

    model_KVnet = model_version(feature_dim=args.feature_dim, cam_intrinsics=dataset.cam_intrinsics,
                                d_candi=d_candi, sigma_soft_max=args.sigma_soft_max, KVNet_feature_dim=args.feature_dim,
                                d_upsample_ratio_KV_net=None, t_win_r=args.t_win, if_refined=args.RNet)

    model_KVnet = torch.nn.DataParallel(model_KVnet, dim=0)
    model_KVnet.cuda(0)

    optimizer_KV = optim.Adam(model_KVnet.parameters(), lr=args.LR, betas=(.9, .999))

    scheduler = optim.lr_scheduler.StepLR(optimizer_KV, step_size=10, gamma=0.1)

    st_epoch = 0
    total_iter = 0
    model_path_KV = args.pre_trained_model_path
    if model_path_KV is not '.':
        print('loading KV_net at %s' % (model_path_KV))
        utils_model.load_pretrained_model(model_KVnet, model_path_KV, optimizer_KV)
        pretrained = torch.load(model_path_KV)
        # st_epoch = 7
        total_iter = pretrained['iter']

    print('Done')

    ##############################################
    # generating volume for depth regression
    dpv_vol = None
    dpv_refined_vol = None
    if dpv_vol is None:
        nDepth, dmap_height, dmap_width = 64, 64, 96
        dpv_vol = torch.ones(1, nDepth, dmap_height, dmap_width).cuda()
        for idepth in range(nDepth):
            dpv_vol[0, idepth, ...] = dpv_vol[0, idepth, ...] * d_candi[idepth]

    if dpv_refined_vol is None:
        nDepth, dmap_height, dmap_width = 64, 256, 384
        dpv_refined_vol = torch.ones(1, nDepth, dmap_height, dmap_width).cuda()
        for idepth in range(nDepth):
            dpv_refined_vol[0, idepth, ...] = dpv_refined_vol[0, idepth, ...] * d_candi[idepth]

    LOSS = []
    for iepoch in range(st_epoch, args.nepoch):
        BatchScheduler = batch_loader.Batch_Loader(
            batch_size=batch_size, fun_get_paths=fun_get_paths,
            dataset_traj=dataset, nTraj=len(traj_Indx), dataset_name=args.dataset)

        for batch_idx in range(len(BatchScheduler)):
            for frame_count, ref_indx in enumerate(range(BatchScheduler.traj_len)):
                local_info = BatchScheduler.local_info()
                n_valid_batch = local_info['is_valid'].sum()

                if n_valid_batch > 0:
                    local_info_valid = batch_loader.get_valid_items(local_info)
                    ref_dats_in = local_info_valid['ref_dats']
                    src_dats_in = local_info_valid['src_dats']
                    cam_intrin_in = local_info_valid['cam_intrins']
                    src_cam_poses_in = torch.cat(local_info_valid['src_cam_poses'], dim=0)

                    if args.pose_noise_level > 0:
                        src_cam_poses_in = add_noise2pose(src_cam_poses_in, args.pose_noise_level)

                    if frame_count == 0 or prev_invalid or no_pred_DPV:
                        prev_invalid = False
                        BVs_predict_in = None
                        print('frame_count ==0 or invalid previous frame or no_pred_DPV version')
                    else:
                        BVs_predict_in = batch_loader.get_valid_BVs(BVs_predict, local_info['is_valid'])

                    BVs_measure, BVs_predict, loss, dmap_log_l, dmap_log_h = train_KVNet_train(
                        n_valid_batch, model_KVnet, optimizer_KV, args.t_win, d_candi,
                        Ref_Dats=ref_dats_in, Src_Dats=src_dats_in, Src_CamPoses=src_cam_poses_in,
                        BVs_predict=BVs_predict_in, Cam_Intrinsics=cam_intrin_in,  dpv_vol=dpv_vol,
                        dpv_refined_vol=dpv_refined_vol, refine_dup=False,
                        weight_var=args.weight_var, loss_type=args.loss_type)

                    loss_v = float(loss.data.cpu().numpy())

                    if n_valid_batch < BatchScheduler.batch_size and not no_pred_DPV:
                        BVs_predict = batch_loader.fill_BVs_predict(BVs_predict, local_info['is_valid'])

                else:
                    loss_v = LOSS[-1]
                    prev_invalid = True

                # Update dat_array #
                if frame_count < BatchScheduler.traj_len - 1:
                    BatchScheduler.proceed_frame()

                total_iter += 1

                # logging #
                if frame_count > 0:
                    LOSS.append(loss_v)
                    print('video batch %d / %d, iter: %d, frame_count: %d; Epoch: %d / %d, loss = %.5f' \
                          % (batch_idx + 1, len(BatchScheduler), total_iter, frame_count, iepoch + 1, args.nepoch,
                             loss_v))

                    var_sum = [var.sum() for var in model_KVnet.parameters() if var.requires_grad]
                    var_cnt = len(var_sum)
                    var_sum = np.sum(var_sum)
                    curr_lr = get_current_lr(optimizer_KV)

                    wandb.log({'loss': float(loss_v),
                               'epoch': iepoch,
                               'model average': var_sum.item() / var_cnt,
                               'lr': curr_lr,
                               }, step=total_iter)

                if total_iter % args.save_model_interv == 0:
                    # if training, save the model #
                    savefilename = saved_model_path + '/' + str(total_iter) + '.tar'
                    torch.save({'iter': total_iter,
                                'frame_count': frame_count,
                                'ref_indx': ref_indx,
                                'traj_idx': batch_idx,
                                'state_dict': model_KVnet.state_dict(),
                                'optimizer': optimizer_KV.state_dict(),
                                'loss': loss_v,
                                'epoch': iepoch}, savefilename)

            BatchScheduler.proceed_batch()
            torch.cuda.empty_cache()
        scheduler.step()

    # stdout.delink()


if __name__ == '__main__':
    main()
