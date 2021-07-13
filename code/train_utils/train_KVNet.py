'''
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license

author: Chao Liu <chaoliu1@cs.cmu.edu>
'''

import numpy as np
import math
import time

import torch

torch.backends.cudnn.benchmark = True
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

import warping.homography as warp_homo
import mutils.misc as m_misc
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

inv_normalize = transforms.Normalize(
    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
)


def train(nGPU, model_KV, optimizer_KV, t_win_r, d_candi, Ref_Dats, Src_Dats, Src_CamPoses,
          BVs_predict, Cam_Intrinsics, refine_dup=False, weight_var=.001, loss_type='NLL', return_confmap_up=False):
    r'''

    Perform one single iteration for the training

    Support multiple GPU traning.
    To do this we treat each trajector as one batch

    Inputs:
    model_KV -
    optimizer_KV -
    Ref_Dats - list of ref_dat

    Src_Dats - list of list of src_dat: [ list_src_dats_traj_0, ...]
                    list_src_dats_traj0[iframe] : NCHW

    Src_CamPoses - N x V x 4 x 4, where N: batch size (# of batched traj), V: # of src. views

    BVs_predict - N x D x H_feat x W_feat

    Cam_Intrinsics - list of camera intrinsics for the batched trajectories

    refine_dup - if upsample the depth dimension in the refinement net

    loss_type = {'L1', 'NLL'}
    L1 - we will calculate the mean value from low res. DPV and filter it with DGF to get the L1 loss in high res.;
         In additional to that, we will also calculate the variance loss
    NLL - we will calulate the NLL loss from the low res. DPV



    Outputs:

    '''

    # prepare for the inputs #
    ref_frame = torch.cat(tuple([ref_dat['img'] for ref_dat in Ref_Dats]), dim=0)
    src_frames_list = [torch.cat(tuple([src_dat_frame['img'] for src_dat_frame in src_dats_traj]), dim=0).unsqueeze(0)
                       for src_dats_traj in Src_Dats]

    src_frames = torch.cat(tuple(src_frames_list), dim=0)
    optimizer_KV.zero_grad()

    # If upsample d in the refinement net#
    if refine_dup:
        dup4_candi = np.linspace(0, d_candi.max(), 4 * len(d_candi))

    # kv-net Forward pass #
    IntMs = torch.cat([cam_intrin['intrinsic_M_cuda'].unsqueeze(0) for cam_intrin in Cam_Intrinsics], dim=0)
    unit_ray_Ms_2D = torch.cat([cam_intrin['unit_ray_array_2D'].unsqueeze(0) for cam_intrin in Cam_Intrinsics],
                               dim=0)

    dmap_cur_refined, dmap_refined, d_dpv, kv_dpv = model_KV(
        ref_frame=ref_frame.cuda(0), src_frames=src_frames.cuda(0), src_cam_poses=Src_CamPoses.cuda(0),
        BV_predict=BVs_predict, IntMs=IntMs.cuda(0), unit_ray_Ms_2D=unit_ray_Ms_2D.cuda(0), mGPU=True,
        cam_intrinsics=Cam_Intrinsics)

    # Get losses #
    loss = 0.
    for ibatch in range(kv_dpv.shape[0]):
        if loss_type == 'NLL':
            # nll loss (d-net) #
            depth_ref = Ref_Dats[ibatch]['dmap'].cuda(kv_dpv.get_device())
            if refine_dup:
                depth_ref_imgsize = Ref_Dats[ibatch]['dmap_up4_imgsize_digit'].cuda(kv_dpv.get_device())
            else:
                depth_ref_imgsize = Ref_Dats[ibatch]['dmap_imgsize_digit'].cuda(kv_dpv.get_device())

            loss = loss + F.nll_loss(d_dpv[ibatch, :, :, :].unsqueeze(0), depth_ref, ignore_index=0)
            loss = loss + F.nll_loss(dmap_cur_refined[ibatch, :, :, :].unsqueeze(0), depth_ref_imgsize, ignore_index=0)

            if BVs_predict is not None:
                if m_misc.valid_dpv(BVs_predict[ibatch, ...]):  # refined
                    loss = loss + F.nll_loss(kv_dpv[ibatch, :, :, :].unsqueeze(0), depth_ref, ignore_index=0)
                    loss = loss + F.nll_loss(dmap_refined[ibatch, :, :, :].unsqueeze(0), depth_ref_imgsize,
                                             ignore_index=0)

            dmap_kv_lowres = m_misc.depth_val_regression(kv_dpv[0, ...].unsqueeze(0), d_candi, BV_log=True)

        else:
            raise Exception('not implemented')

    # Backward pass #
    loss = loss / torch.tensor(nGPU.astype(np.float)).cuda(loss.get_device())
    loss.backward()
    optimizer_KV.step()

    # BV_predict estimation (3D re-sampling) #
    d_dpv = d_dpv.detach()
    kv_dpv = kv_dpv.detach()
    r_dpv = dmap_cur_refined.detach() if dmap_cur_refined is not -1 else dmap_refined.detach()
    BVs_predict_out = []

    for ibatch in range(d_dpv.shape[0]):
        rel_Rt = Src_CamPoses[ibatch, t_win_r, :, :].inverse()
        BV_predict = warp_homo.resample_vol_cuda(src_vol=kv_dpv[ibatch, ...].unsqueeze(0),
                                                 rel_extM=rel_Rt.cuda(kv_dpv.get_device()),
                                                 cam_intrinsic=Cam_Intrinsics[ibatch],
                                                 d_candi=d_candi,
                                                 padding_value=math.log(1. / float(len(d_candi)))
                                                 ).clamp(max=0, min=-1000.).unsqueeze(0)
        BVs_predict_out.append(BV_predict)

    BVs_predict_out = torch.cat(BVs_predict_out, dim=0)

    # logging (for single GPU) #
    depth_ref_lowres = Ref_Dats[0]['dmap_raw'].cpu().squeeze().numpy()
    depth_kv_lres_log = dmap_kv_lowres[0, ...].detach().cpu().squeeze().numpy()
    dmap_log_lres = np.hstack([depth_kv_lres_log, depth_ref_lowres])
    if refine_dup:
        depth_kv_hres_log = m_misc.depth_val_regression(
            dmap_refined[0, ...].unsqueeze(0), dup4_candi, BV_log=True).detach().cpu().squeeze().numpy()

    else:
        depth_kv_hres_log = m_misc.depth_val_regression(
            dmap_refined[0, ...].unsqueeze(0), d_candi, BV_log=True).detach().cpu().squeeze().numpy()

    depth_ref_imgsize_raw = Ref_Dats[0]['dmap_imgsize'].squeeze().cpu().numpy()
    dmap_log_hres = np.hstack([depth_kv_hres_log, depth_ref_imgsize_raw])

    if return_confmap_up:
        confmap_up = torch.exp(dmap_refined[0, ...].detach())
        confmap_up, _ = torch.max(confmap_up, dim=0)
        return r_dpv, BVs_predict_out, loss, dmap_log_lres, dmap_log_hres, confmap_up.cpu()

    else:
        return r_dpv, BVs_predict_out, loss, dmap_log_lres, dmap_log_hres


def train_no_pred_DPV(nGPU, model_KV, optimizer_KV, t_win_r, d_candi, Ref_Dats, Src_Dats, Src_CamPoses,
                      BVs_predict, Cam_Intrinsics, dpv_vol, dpv_refined_vol,
                      refine_dup=False, weight_var=.001, loss_type='NLL', return_confmap_up=False):
    r'''

    Perform one single iteration for the training

    Support multiple GPU traning.
    To do this we treat each trajector as one batch

    Inputs:
    model_KV -
    optimizer_KV -
    Ref_Dats - list of ref_dat

    Src_Dats - list of list of src_dat: [ list_src_dats_traj_0, ...]
                    list_src_dats_traj0[iframe] : NCHW

    Src_CamPoses - N x V x 4 x 4, where N: batch size (# of batched traj), V: # of src. views

    BVs_predict - N x D x H_feat x W_feat

    Cam_Intrinsics - list of camera intrinsics for the batched trajectories

    refine_dup - if upsample the depth dimension in the refinement net

    loss_type = {'L1', 'NLL'}
    L1 - we will calculate the mean value from low res. DPV and filter it with DGF to get the L1 loss in high res.;
         In additional to that, we will also calculate the variance loss
    NLL - we will calulate the NLL loss from the low res. DPV

    Outputs:

    '''

    # prepare for the inputs #
    ref_frame = torch.cat(tuple([ref_dat['img'] for ref_dat in Ref_Dats]), dim=0)
    src_frames_list = [torch.cat(tuple([src_dat_frame['img'] for src_dat_frame in src_dats_traj]), dim=0).unsqueeze(0)
                       for src_dats_traj in Src_Dats]

    src_frames = torch.cat(tuple(src_frames_list), dim=0)
    optimizer_KV.zero_grad()

    # If upsample d in the refinement net#
    if refine_dup:
        dup4_candi = np.linspace(0, d_candi.max(), 4 * len(d_candi))

    # kv-net Forward pass #
    IntMs = torch.cat([intrin['intrinsic_M_cuda'].unsqueeze(0) for intrin in Cam_Intrinsics], dim=0)
    unit_ray_Ms_2D = torch.cat([intrin['unit_ray_array_2D'].unsqueeze(0) for intrin in Cam_Intrinsics], dim=0)

    BatchIdx = torch.IntTensor(np.arange(nGPU))
    dmap_cur_refined, dmap_refined, d_dpv, kv_dpv = model_KV(
        ref_frame=ref_frame.cuda(0), src_frames=src_frames.cuda(0), src_cam_poses=Src_CamPoses.cuda(0),
        BV_predict=BVs_predict, IntMs=IntMs.cuda(0), unit_ray_Ms_2D=unit_ray_Ms_2D.cuda(0), mGPU=True,
        cam_intrinsics=Cam_Intrinsics, BatchIdx=BatchIdx)

    # Get losses #
    loss = 0.

    for ibatch in range(kv_dpv.shape[0]):
        if loss_type == 'silog':
            depth_ref = Ref_Dats[ibatch]['dmap_raw'].cuda(d_dpv.get_device())  ##FloatTensor
            # depth_ref = Ref_Dats[ibatch]['dmap'].cuda(d_dpv.get_device())

            if refine_dup:
                depth_ref_imgsize = Ref_Dats[ibatch]['dmap_imgsize'].cuda(d_dpv.get_device())
                # depth_ref_imgsize = Ref_Dats[ibatch]['dmap_up4_imgsize_digit'].cuda(d_dpv.get_device())
            else:
                depth_ref_imgsize = Ref_Dats[ibatch]['dmap_imgsize'].cuda(d_dpv.get_device())
                # depth_ref_imgsize = Ref_Dats[ibatch]['dmap_imgsize_digit'].cuda(d_dpv.get_device())

            d_dpv_depth = torch.sum(
                (torch.exp(d_dpv[ibatch]) * dpv_vol.squeeze(0)), dim=0).squeeze(0)
            dmap_cur_refined_depth = torch.sum(
                (torch.exp(dmap_cur_refined[ibatch]) * dpv_refined_vol.squeeze(0)), dim=0).squeeze(0)
            kv_dpv_depth = torch.sum(
                (torch.exp(kv_dpv[ibatch]) * dpv_vol.squeeze(0)), dim=0).squeeze(0)
            dmap_refined_depth = torch.sum(
                (torch.exp(dmap_refined[ibatch]) * dpv_refined_vol.squeeze(0)), dim=0).squeeze(0)

            depth_ref = depth_ref.squeeze(0)
            depth_ref_imgsize = depth_ref_imgsize.squeeze(0)
            mask_dpv = ((1.0 <= depth_ref) & (depth_ref <= len(d_candi)))
            mask_dpv_refined = ((1.0 <= depth_ref_imgsize) & (depth_ref_imgsize <= len(d_candi)))

            loss = loss + silog_loss(d_dpv_depth, depth_ref, mask_dpv)
            loss = loss + silog_loss(dmap_cur_refined_depth, depth_ref_imgsize, mask_dpv_refined)
            loss = loss + silog_loss(kv_dpv_depth, depth_ref, mask_dpv)
            loss = loss + silog_loss(dmap_refined_depth, depth_ref_imgsize, mask_dpv_refined)

            dmap_kv_lowres = m_misc.depth_val_regression(kv_dpv[0].unsqueeze(0), d_candi, BV_log=True)



        elif loss_type == 'NLL':
            # nll loss (d-net) #
            depth_ref = Ref_Dats[ibatch]['dmap'].cuda(kv_dpv.get_device())
            if refine_dup:
                depth_ref_imgsize = Ref_Dats[ibatch]['dmap_up4_imgsize_digit'].cuda(kv_dpv.get_device())
            else:
                depth_ref_imgsize = Ref_Dats[ibatch]['dmap_imgsize_digit'].cuda(kv_dpv.get_device())

            loss = loss + F.nll_loss(d_dpv[ibatch, :, :, :].unsqueeze(0), depth_ref, ignore_index=0)
            loss = loss + F.nll_loss(dmap_cur_refined[ibatch, :, :, :].unsqueeze(0), depth_ref_imgsize, ignore_index=0)
            loss = loss + F.nll_loss(kv_dpv[ibatch, :, :, :].unsqueeze(0), depth_ref, ignore_index=0)
            loss = loss + F.nll_loss(dmap_refined[ibatch, :, :, :].unsqueeze(0), depth_ref_imgsize, ignore_index=0)

            dmap_kv_lowres = m_misc.depth_val_regression(kv_dpv[0].unsqueeze(0), d_candi, BV_log=True)

        else:
            raise Exception('not implemented')

    # Backward pass #
    loss = loss / torch.tensor(nGPU.astype(np.float)).cuda(loss.get_device())
    loss.backward()
    optimizer_KV.step()

    # BV_predict estimation (3D re-sampling) #
    r_dpv = dmap_cur_refined.detach() if dmap_cur_refined is not -1 else dmap_refined.detach()
    BVs_predict_out = None

    # logging (for single GPU) #
    depth_ref_lowres = Ref_Dats[0]['dmap_raw'].cpu().squeeze().numpy()
    depth_kv_lres_log = dmap_kv_lowres[0, ...].detach().cpu().squeeze().numpy()
    dmap_log_lres = np.hstack([depth_kv_lres_log, depth_ref_lowres])

    if refine_dup:
        depth_kv_hres_log = m_misc.depth_val_regression(
            dmap_refined[0, ...].unsqueeze(0), dup4_candi, BV_log=True).detach().cpu().squeeze().numpy()

    else:
        depth_kv_hres_log = m_misc.depth_val_regression(
            dmap_refined[0, ...].unsqueeze(0), d_candi, BV_log=True).detach().cpu().squeeze().numpy()

    depth_ref_imgsize_raw = Ref_Dats[0]['dmap_imgsize'].squeeze().cpu().numpy()
    dmap_log_hres = np.hstack([depth_kv_hres_log, depth_ref_imgsize_raw])

    if return_confmap_up:
        confmap_up = torch.exp(dmap_refined[0, ...].detach())
        confmap_up, _ = torch.max(confmap_up, dim=0)
        return r_dpv, BVs_predict_out, loss, dmap_log_lres, dmap_log_hres, confmap_up.cpu()

    else:
        return r_dpv, BVs_predict_out, loss, dmap_log_lres, dmap_log_hres


def train_per_frame(nGPU, model_KV, optimizer_KV, t_win_r, d_candi, Ref_Dats, Src_Dats, Src_CamPoses,
                    BVs_predict, Cam_Intrinsics, dpv_vol, dpv_refined_vol,
                    refine_dup=False, weight_var=.001, loss_type='NLL', return_confmap_up=False):
    r'''

    Perform one single iteration for the training 

    Support multiple GPU traning. 
    To do this we treat each trajector as one batch

    Inputs: 
    model_KV - 
    optimizer_KV - 
    Ref_Dats - list of ref_dat 

    Src_Dats - list of list of src_dat: [ list_src_dats_traj_0, ...]
                    list_src_dats_traj0[iframe] : NCHW

    Src_CamPoses - N x V x 4 x 4, where N: batch size (# of batched traj), V: # of src. views

    BVs_predict - N x D x H_feat x W_feat

    Cam_Intrinsics - list of camera intrinsics for the batched trajectories

    refine_dup - if upsample the depth dimension in the refinement net

    loss_type = {'L1', 'NLL'}
    L1 - we will calculate the mean value from low res. DPV and filter it with DGF to get the L1 loss in high res.; 
         In additional to that, we will also calculate the variance loss
    NLL - we will calulate the NLL loss from the low res. DPV

    Outputs:

    '''

    # prepare for the inputs #
    ref_frame = torch.cat(tuple([ref_dat['img'] for ref_dat in Ref_Dats]), dim=0)
    optimizer_KV.zero_grad()

    # kv-net Forward pass #
    dmap_cur_refined, dmap_refined, d_dpv, kv_dpv = model_KV(ref_frame=ref_frame.cuda(0))
    # Get losses #
    loss = 0.
    # dpv_vol = None
    # dpv_refined_vol = None
    # per-frame approach
    for ibatch in range(d_dpv.shape[0]):
        if loss_type == 'silog':
            depth_ref = Ref_Dats[ibatch]['dmap_raw'].cuda(d_dpv.get_device())  ##FloatTensor
            if refine_dup:
                depth_ref_imgsize = Ref_Dats[ibatch]['dmap_imgsize'].cuda(d_dpv.get_device())
            else:
                depth_ref_imgsize = Ref_Dats[ibatch]['dmap_imgsize'].cuda(d_dpv.get_device())

            d_dpv_depth = torch.sum(
                (torch.exp(d_dpv[ibatch]) * dpv_vol.squeeze(0)), dim=0).squeeze(0)
            dmap_cur_refined_depth = torch.sum(
                (torch.exp(dmap_cur_refined[ibatch]) * dpv_refined_vol.squeeze(0)), dim=0).squeeze(0)

            depth_ref = depth_ref.squeeze(0)
            depth_ref_imgsize = depth_ref_imgsize.squeeze(0)
            mask_dpv = ((1.0 <= depth_ref) & (depth_ref <= len(d_candi)))
            mask_dpv_refined = ((1.0 <= depth_ref_imgsize) & (depth_ref_imgsize <= len(d_candi)))

            loss = loss + silog_loss(d_dpv_depth, depth_ref, mask_dpv)
            loss = loss + silog_loss(dmap_cur_refined_depth, depth_ref_imgsize, mask_dpv_refined)

        elif loss_type == 'NLL':
            # nll loss (d-net) #

            depth_ref = Ref_Dats[ibatch]['dmap'].cuda(d_dpv.get_device())
            if refine_dup:
                depth_ref_imgsize = Ref_Dats[ibatch]['dmap_up4_imgsize_digit'].cuda(d_dpv.get_device())
            else:
                depth_ref_imgsize = Ref_Dats[ibatch]['dmap_imgsize_digit'].cuda(d_dpv.get_device())

            loss = loss + F.nll_loss(d_dpv[ibatch, :, :, :].unsqueeze(0), depth_ref, ignore_index=0)
            loss = loss + F.nll_loss(dmap_cur_refined[ibatch, :, :, :].unsqueeze(0), depth_ref_imgsize, ignore_index=0)

    loss = loss / torch.tensor(nGPU.astype(np.float)).cuda(loss.get_device())
    loss.backward()
    optimizer_KV.step()

    BVs_predict_out = None
    r_dpv = None
    dmap_log_lres = None
    dmap_log_hres = None

    # logging (for single GPU) #
    if return_confmap_up:
        confmap_up = torch.exp(dmap_refined[0, ...].detach())
        confmap_up, _ = torch.max(confmap_up, dim=0)

        return r_dpv, BVs_predict_out, loss, dmap_log_lres, dmap_log_hres, confmap_up.cpu()

    else:
        return r_dpv, BVs_predict_out, loss, dmap_log_lres, dmap_log_hres


def silog_loss(depth_est, depth_gt, mask):
    d = torch.log(depth_est[mask]) - torch.log(depth_gt[mask])
    return torch.sqrt((d ** 2).mean() - 0.85 * (d.mean() ** 2)) * 10.0
