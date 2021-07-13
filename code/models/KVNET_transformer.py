'''
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license

author: Chao Liu <chaoliu1@cs.cmu.edu>
'''

'''
The full KV-net framework,
support (still in progress) multiple-gpu training
'''

import torch

import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable
from models.psm_submodule import *
import warping.homography as warp_homo
import math
import time

import models.basic as model_basic
import models.Refine as model_refine
import models.non_local_block as model_transformer

import mutils.misc as m_misc
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from mdataloader.m_preprocess import UnNormalize


# ver1-per_frame
class KVNET1(nn.Module):
    def __init__(self, feature_dim, cam_intrinsics, d_candi, sigma_soft_max,
                 KVNet_feature_dim, d_upsample_ratio_KV_net,
                 if_refined=True, refineNet_name='DPV',
                 t_win_r=2, refine_channel=3, if_upsample_d=False):
        super(KVNET1, self).__init__()
        self.t_win_r = t_win_r
        self.feature_dim = feature_dim
        self.KVNet_feature_dim = KVNet_feature_dim
        self.sigma_soft_max = sigma_soft_max
        self.d_upsample_ratio_KV_net = d_upsample_ratio_KV_net
        self.d_candi = d_candi
        self.if_refined = if_refined
        self.refineNet_name = refineNet_name
        self.if_upsample_d = if_upsample_d

        # submodules #
        if self.if_refined and self.refineNet_name == 'DPV':
            self.feature_extractor = model_basic.feature_extractor(feature_dim=feature_dim, multi_scale=True)

        else:
            self.feature_extractor = model_basic.feature_extractor(feature_dim=feature_dim)

        if self.if_refined:
            if self.refineNet_name == 'DGF':
                self.r_net = model_refine.RefineNet_DGF(refine_channel)
            elif self.refineNet_name == 'DPV':
                self.r_net = model_refine.RefineNet_DPV_upsample(
                    int(self.feature_dim), int(self.feature_dim / 2), 3,
                    D=len(self.d_candi), upsample_D=self.if_upsample_d)

        # print #
        print('KV-Net initialization:')
        print('with R-net: %r' % (self.if_refined))
        if self.if_refined:
            print('\trefinement name: %s' % (self.refineNet_name))

    def forward(self, ref_frame, src_frames=None, src_cam_poses=None, BV_predict=None, IntMs=None, unit_ray_Ms_2D=None,
                mGPU=False, BatchIdx=None, cam_intrinsics=None):
        """
        ver1-per_frame
        """
        # D-Net #
        if (self.if_refined is False) or (self.if_refined is True and self.refineNet_name != 'DPV'):
            raise NotImplementedError

        else:
            feat_imgs_layer_1, feat_imgs = self.feature_extractor(ref_frame)
            feat_img_ref_layer1 = feat_imgs_layer_1#[-1, ...].unsqueeze(0)
            feat_img_ref = feat_imgs#[-1, ...].unsqueeze(0)  # [B, D, H/4, W/4]
            d_net_features = [feat_img_ref, feat_img_ref_layer1, ref_frame]
            BV_cur = F.log_softmax(feat_img_ref, dim=1)  # [B, D, H/4, W/4]

        if self.if_refined:
            if self.refineNet_name == 'DGF':
                raise NotImplementedError
            elif self.refineNet_name == 'DPV':
                dmap_cur_refined = self.r_net(torch.exp(BV_cur), img_features=d_net_features)  # [1, D, H, W]
        else:
            dmap_cur_refined = -1

        return dmap_cur_refined, dmap_cur_refined, BV_cur, BV_cur


# ver2-SA_pred
class KVNET2(nn.Module):
    r'''
    Inside this module, we will do the full KV-Net pipeline:
    * D-Net (feature extractiion + BV_cur estimation )
    * KV-Net
    '''

    """
    model_KVnet = m_kvnet.KVNET(feature_dim=args.feature_dim, cam_intrinsics=dataset.cam_intrinsics, d_candi=d_candi, sigma_soft_max=args.sigma_soft_max, 
                                KVNet_feature_dim=args.feature_dim, d_upsample_ratio_KV_net=None,
                                if_refined=args.RNet, 
                                t_win_r=args.t_win)
    """

    def __init__(self, feature_dim, cam_intrinsics, d_candi, sigma_soft_max,
                 KVNet_feature_dim, d_upsample_ratio_KV_net,
                 if_refined=True, refineNet_name='DPV',
                 t_win_r=2, refine_channel=3, if_upsample_d=False):
        r'''
        inputs:
        refineNet_name - {'DGF', 'DPV'}
        if_refined - if use the refinement net for upsampling
        refine_channel - the # of channels for the guided image in the refinement net, by default 3 for rgb input image
        '''

        super(KVNET2, self).__init__()
        self.t_win_r = t_win_r
        self.feature_dim = feature_dim
        self.KVNet_feature_dim = KVNet_feature_dim
        self.sigma_soft_max = sigma_soft_max
        self.d_upsample_ratio_KV_net = d_upsample_ratio_KV_net
        self.d_candi = d_candi
        self.if_refined = if_refined
        self.refineNet_name = refineNet_name
        self.if_upsample_d = if_upsample_d

        # submodules #
        if self.if_refined and self.refineNet_name == 'DPV':
            self.feature_extractor = model_basic.feature_extractor(feature_dim=feature_dim, multi_scale=True)
            self.d_net = model_basic.D_NET_BASIC(
                self.feature_extractor, cam_intrinsics, d_candi,
                sigma_soft_max, BV_log=True, normalize=True,
                use_img_intensity=True, force_img_dw_rate=1,
                parallel_d=True, output_features=True,
                refine_costV=False, feat_dist='L2')

        else:
            self.feature_extractor = model_basic.feature_extractor(feature_dim=feature_dim)
            self.d_net = model_basic.D_NET_BASIC(
                self.feature_extractor, cam_intrinsics,
                d_candi, sigma_soft_max, use_img_intensity=True,
                BV_log=True, output_features=False)

        self.kv_net = model_basic.KV_NET_BASIC(3 * (t_win_r * 2 + 1) + 1,
                                               feature_dim=KVNet_feature_dim,
                                               up_sample_ratio=d_upsample_ratio_KV_net)

        self.transformer1 = model_transformer.NONLocalBlock2D(KVNet_feature_dim, sub_sample=False, bn_layer=True)
        self.transformer2 = model_transformer.NONLocalBlock2D(KVNet_feature_dim, sub_sample=False, bn_layer=True)
        if self.if_refined:
            if self.refineNet_name == 'DGF':
                self.r_net = model_refine.RefineNet_DGF(refine_channel)
            elif self.refineNet_name == 'DPV':
                self.r_net = model_refine.RefineNet_DPV_upsample(
                    int(self.feature_dim), int(self.feature_dim / 2), 3,
                    D=len(self.d_candi), upsample_D=self.if_upsample_d)

        # print #
        print('KV-Net initialization:')
        print('with R-net: %r' % (self.if_refined))
        if self.if_refined:
            print('\trefinement name: %s' % (self.refineNet_name))

    def forward(self, ref_frame, src_frames, src_cam_poses, BV_predict=None, IntMs=None, unit_ray_Ms_2D=None,
                mGPU=False, BatchIdx=None, cam_intrinsics=None):
        r'''
        ver2-SA_pred
        '''
        if isinstance(BV_predict, torch.Tensor):
            if m_misc.valid_dpv(BV_predict):
                assert BV_predict.shape[0] == 1

        # D-Net #
        if (self.if_refined is False) or (self.if_refined is True and self.refineNet_name != 'DPV'):
            BV_cur = self.d_net(ref_frame, src_frames, src_cam_poses, BV_predict=None, debug_ipdb=False)

        else:
            # here
            BV_cur, d_net_features = self.d_net(ref_frame, src_frames, src_cam_poses, BV_predict=None, debug_ipdb=False)
            d_net_features.append(ref_frame)

        # transformer!
        BV_cur = self.transformer1(BV_cur)
        BV_cur = F.log_softmax(BV_cur, dim=1)

        if self.if_refined:
            if self.refineNet_name == 'DGF':
                raise NotImplementedError
            elif self.refineNet_name == 'DPV':
                dmap_cur_refined = self.r_net(torch.exp(BV_cur), img_features=d_net_features)  # [1, D, H, W]
        else:
            dmap_cur_refined = -1

        if not isinstance(BV_predict, torch.Tensor):
            # If the first time win., then return only BV_cur
            return dmap_cur_refined, dmap_cur_refined, BV_cur, BV_cur

        elif not m_misc.valid_dpv(BV_predict):
            return dmap_cur_refined, dmap_cur_refined, BV_cur, BV_cur

        else:
            # KV-Net #
            down_sample_rate = ref_frame.shape[3] / BV_cur.shape[3]

            ref_frame_dw = F.avg_pool2d(ref_frame, int(down_sample_rate)).cuda()
            src_frames_dw = [F.avg_pool2d(src_frame_.unsqueeze(0), int(down_sample_rate)).cuda()
                             for src_frame_ in src_frames.squeeze(0)]

            Rs_src = [pose[:3, :3] for pose in src_cam_poses.squeeze(0)]
            ts_src = [pose[:3, 3] for pose in src_cam_poses.squeeze(0)]

            # Warp the src-frames to the ref. view #
            if mGPU:
                WAPRED_src_frames = warp_homo.warp_img_feats_mgpu(
                    src_frames_dw, self.d_candi, Rs_src, ts_src, IntMs, unit_ray_Ms_2D)
            else:
                cam_intrin = cam_intrinsics[int(BatchIdx)]
                WAPRED_src_frames = warp_homo.warp_img_feats_v3(src_frames_dw, self.d_candi, Rs_src, ts_src,
                                                                cam_intrin, )

            ref_frame_dw_rep = torch.transpose(ref_frame_dw.repeat([len(self.d_candi), 1, 1, 1]), 0, 1)

            # Input to the KV-net #
            # torch.cat(tuple(WAPRED_src_frames), dim=0).shape: [3*4, D, H/4, W/4]
            # ref_frame_dw_rep.shape: [3, D, H/4, W/4]
            # (BV_cur - BV_predict).shape: [1, D, H/4, W/4]
            # kvnet_in_vol.shape: [1, 16, D, H/4, W/4]

            kvnet_in_vol = torch.cat(
                (torch.cat(tuple(WAPRED_src_frames), dim=0), ref_frame_dw_rep, BV_cur - BV_predict), dim=0).unsqueeze(0)

            # Run KV-net #
            BV_gain = self.kv_net(kvnet_in_vol)

            # Add back to BV_predict #
            DPV = torch.squeeze(BV_gain, dim=1) + BV_predict
            # transformer!
            DPV = self.transformer2(DPV)
            DPV = F.log_softmax(DPV, dim=1)
            if self.if_refined:
                if self.refineNet_name == 'DGF':
                    dmap_lowres = m_misc.depth_val_regression(DPV, self.d_candi, BV_log=True).unsqueeze(0)
                    dmap_refined = self.r_net(dmap_lowres, ref_frame)
                elif self.refineNet_name == 'DPV':
                    dmap_refined = self.r_net(torch.exp(DPV), img_features=d_net_features)
            else:
                dmap_refined = -1

            return dmap_cur_refined, dmap_refined, BV_cur, DPV


# ver3-SA_NO_pred
class KVNET3(nn.Module):
    r'''
    Inside this module, we will do the full KV-Net pipeline:
    * D-Net (feature extractiion + BV_cur estimation )
    * KV-Net
    '''

    def __init__(self, feature_dim, cam_intrinsics, d_candi, sigma_soft_max,
                 KVNet_feature_dim, d_upsample_ratio_KV_net,
                 if_refined=True, refineNet_name='DPV',
                 t_win_r=2, refine_channel=3, if_upsample_d=False):
        r'''
        inputs:
        refineNet_name - {'DGF', 'DPV'}
        if_refined - if use the refinement net for upsampling
        refine_channel - the # of channels for the guided image in the refinement net, by default 3 for rgb input image
        '''

        super(KVNET3, self).__init__()
        self.t_win_r = t_win_r
        self.feature_dim = feature_dim
        self.KVNet_feature_dim = KVNet_feature_dim
        self.sigma_soft_max = sigma_soft_max
        self.d_upsample_ratio_KV_net = d_upsample_ratio_KV_net
        self.d_candi = d_candi
        self.if_refined = if_refined
        self.refineNet_name = refineNet_name
        self.if_upsample_d = if_upsample_d

        # submodules #
        if self.if_refined and self.refineNet_name == 'DPV':
            self.feature_extractor = model_basic.feature_extractor(feature_dim=feature_dim, multi_scale=True)
            self.d_net = model_basic.D_NET_BASIC(
                self.feature_extractor, cam_intrinsics, d_candi,
                sigma_soft_max, BV_log=True, normalize=True,
                use_img_intensity=True, force_img_dw_rate=1,
                parallel_d=True, output_features=True,
                refine_costV=False, feat_dist='L2')

        else:
            self.feature_extractor = model_basic.feature_extractor(feature_dim=feature_dim)
            self.d_net = model_basic.D_NET_BASIC(
                self.feature_extractor, cam_intrinsics,
                d_candi, sigma_soft_max, use_img_intensity=True,
                BV_log=True, output_features=False)

        self.kv_net = model_basic.KV_NET_BASIC(3 * (t_win_r * 2 + 1) + 1,
                                               feature_dim=KVNet_feature_dim,
                                               up_sample_ratio=d_upsample_ratio_KV_net)
        self.transformer1 = model_transformer.NONLocalBlock2D(KVNet_feature_dim, sub_sample=False, bn_layer=True)
        self.transformer2 = model_transformer.NONLocalBlock2D(KVNet_feature_dim, sub_sample=False, bn_layer=True)
        if self.if_refined:
            if self.refineNet_name == 'DGF':
                self.r_net = model_refine.RefineNet_DGF(refine_channel)
            elif self.refineNet_name == 'DPV':
                self.r_net = model_refine.RefineNet_DPV_upsample(
                    int(self.feature_dim), int(self.feature_dim / 2), 3,
                    D=len(self.d_candi), upsample_D=self.if_upsample_d)

        # print #
        print('KV-Net initialization:')
        print('with R-net: %r' % (self.if_refined))
        if self.if_refined:
            print('\trefinement name: %s' % (self.refineNet_name))

    def forward(self, ref_frame, src_frames, src_cam_poses, BV_predict=None, IntMs=None, unit_ray_Ms_2D=None,
                mGPU=False, BatchIdx=None, cam_intrinsics=None):
        r'''
        ver3-SA_NO_pred
        '''
        if isinstance(BV_predict, torch.Tensor):
            if m_misc.valid_dpv(BV_predict):
                assert BV_predict.shape[0] == 1

        # D-Net #
        if (self.if_refined is False) or (self.if_refined is True and self.refineNet_name != 'DPV'):
            raise NotImplementedError

        else:
            # here
            DPV, d_net_features = self.d_net(ref_frame, src_frames, src_cam_poses, BV_predict=None, debug_ipdb=False)
            d_net_features.append(ref_frame)
            feat_img_ref = d_net_features[0]
            BV_cur = F.log_softmax(feat_img_ref, dim=1)

        # transformer!

        DPV = self.transformer1(DPV)
        DPV = F.log_softmax(DPV, dim=1)

        if isinstance(BV_predict, torch.Tensor):
            raise Exception("should be None")

        else:
            # KV-Net #
            down_sample_rate = ref_frame.shape[3] / BV_cur.shape[3]

            ref_frame_dw = F.avg_pool2d(ref_frame, int(down_sample_rate)).cuda()
            src_frames_dw = [F.avg_pool2d(src_frame_.unsqueeze(0), int(down_sample_rate)).cuda()
                             for src_frame_ in src_frames.squeeze(0)]

            Rs_src = [pose[:3, :3] for pose in src_cam_poses.squeeze(0)]
            ts_src = [pose[:3, 3] for pose in src_cam_poses.squeeze(0)]

            # Warp the src-frames to the ref. view #
            if mGPU:
                WAPRED_src_frames = warp_homo.warp_img_feats_mgpu(
                    src_frames_dw, self.d_candi, Rs_src, ts_src, IntMs, unit_ray_Ms_2D)
            else:
                cam_intrin = cam_intrinsics[int(BatchIdx)]
                WAPRED_src_frames = warp_homo.warp_img_feats_v3(src_frames_dw, self.d_candi, Rs_src, ts_src,
                                                                cam_intrin, )

            ref_frame_dw_rep = torch.transpose(ref_frame_dw.repeat([len(self.d_candi), 1, 1, 1]), 0, 1)

            # Input to the KV-net #
            # torch.cat(tuple(WAPRED_src_frames), dim=0).shape: [3*4, D, H/4, W/4]
            # ref_frame_dw_rep.shape: [3, D, H/4, W/4]
            # (BV_cur - BV_predict).shape: [1, D, H/4, W/4]
            # kvnet_in_vol.shape: [1, 16, D, H/4, W/4]

            kvnet_in_vol = torch.cat(
                (torch.cat(tuple(WAPRED_src_frames), dim=0), ref_frame_dw_rep, DPV), dim=0).unsqueeze(0)

            # Run KV-net #
            BV_gain = self.kv_net(kvnet_in_vol)

            # Add back to BV_predict #
            DPV = torch.squeeze(BV_gain, dim=1)
            # transformer!
            DPV = self.transformer2(DPV)
            DPV = F.log_softmax(DPV, dim=1)
            if self.if_refined:
                if self.refineNet_name == 'DGF':
                    dmap_lowres = m_misc.depth_val_regression(DPV, self.d_candi, BV_log=True).unsqueeze(0)
                    dmap_refined = self.r_net(dmap_lowres, ref_frame)
                elif self.refineNet_name == 'DPV':
                    dmap_refined = self.r_net(torch.exp(DPV), img_features=d_net_features)
                    dmap_cur_refined = self.r_net(torch.exp(BV_cur), img_features=d_net_features)  # [1, D, H, W]

            else:
                dmap_refined = -1

            return dmap_cur_refined, dmap_refined, BV_cur, DPV


# ver4-aggr
class KVNET4(nn.Module):
    def __init__(self, feature_dim, cam_intrinsics, d_candi, sigma_soft_max,
                 KVNet_feature_dim, d_upsample_ratio_KV_net,
                 if_refined=True, refineNet_name='DPV',
                 t_win_r=2, refine_channel=3, if_upsample_d=False):

        super(KVNET4, self).__init__()
        self.t_win_r = t_win_r
        self.feature_dim = feature_dim
        self.KVNet_feature_dim = KVNet_feature_dim
        self.sigma_soft_max = sigma_soft_max
        self.d_upsample_ratio_KV_net = d_upsample_ratio_KV_net
        self.d_candi = d_candi
        self.if_refined = if_refined
        self.refineNet_name = refineNet_name
        self.if_upsample_d = if_upsample_d

        # submodules #
        if self.if_refined and self.refineNet_name == 'DPV':
            self.feature_extractor = model_basic.feature_extractor(feature_dim=feature_dim, multi_scale=True)


        else:
            self.feature_extractor = model_basic.feature_extractor(feature_dim=feature_dim)

        self.kv_net = model_basic.KV_NET_BASIC(3 * (t_win_r * 2 + 1) + 1,
                                               feature_dim=KVNet_feature_dim,
                                               up_sample_ratio=d_upsample_ratio_KV_net)

        if self.if_refined:
            if self.refineNet_name == 'DGF':
                self.r_net = model_refine.RefineNet_DGF(refine_channel)
            elif self.refineNet_name == 'DPV':
                self.r_net = model_refine.RefineNet_DPV_upsample(
                    int(self.feature_dim), int(self.feature_dim / 2), 3,
                    D=len(self.d_candi), upsample_D=self.if_upsample_d)
        self.transformer_3D = model_transformer.NONLocalBlock3D(KVNet_feature_dim, sub_sample=False, bn_layer=True,
                                                                ref=True)

        # print #
        print('KV-Net initialization:')
        print('with R-net: %r' % (self.if_refined))
        if self.if_refined:
            print('\trefinement name: %s' % (self.refineNet_name))

    def forward(self, ref_frame, src_frames, src_cam_poses, cam_intrinsics, BV_predict=None, IntMs=None,
                unit_ray_Ms_2D=None, mGPU=False, BatchIdx=None):
        r'''
        ver4-aggr
        '''

        # D-Net #
        if (self.if_refined is False) or (self.if_refined is True and self.refineNet_name != 'DPV'):
            raise NotImplementedError

        else:
            # here
            feat_imgs_layer_1, feat_imgs = self.feature_extractor(torch.cat((src_frames[0, ...], ref_frame), dim=0))
            feat_img_ref_layer1 = feat_imgs_layer_1[-1, ...].unsqueeze(0)
            feat_img_ref = feat_imgs[-1, ...].unsqueeze(0)  # [1, D, H/4, W/4]
            feat_imgs_src = feat_imgs[:-1, ...].unsqueeze(0)  # [1, 4, D, H/4, W/4]
            d_net_features = [feat_img_ref, feat_img_ref_layer1, ref_frame]

        # warp src features to ref view
        warped_src_feats = []
        N = 0
        for src_num in range(feat_imgs_src.shape[1]):
            rel_Rt = src_cam_poses[N, src_num, :, :]
            warped_src_feat = warp_homo.resample_vol_cuda(
                src_vol=F.log_softmax(feat_imgs_src[:, src_num].clone(), dim=1),
                rel_extM=rel_Rt.cuda(feat_imgs_src.get_device()),
                cam_intrinsic=cam_intrinsics[BatchIdx[0].item()],
                d_candi=self.d_candi,
                padding_value=math.log(1. / float(len(self.d_candi)))
            ).clamp(max=0, min=-1000.).unsqueeze(0)
            warped_src_feats.append(warped_src_feat)

        warped_src_feats = torch.cat(warped_src_feats, dim=0)
        feat_imgs[:-1] = warped_src_feats
        feat_imgs = F.log_softmax(feat_imgs, dim=1)  # logSM version
        BV_cur = F.log_softmax(feat_img_ref, dim=1)


        # Temporal Attention / input: [b,c,t,h,w]
        DPV = self.transformer_3D.forward_ref(torch.exp(feat_imgs).unsqueeze(0).permute(0, 2, 1, 3, 4), ref_idx=-1)

        if isinstance(BV_predict, torch.Tensor):
            raise Exception("BV_predict should be None!")

        else:
            # KV-Net #
            down_sample_rate = ref_frame.shape[3] / BV_cur.shape[3]

            ref_frame_dw = F.avg_pool2d(ref_frame, int(down_sample_rate)).cuda()
            src_frames_dw = [F.avg_pool2d(src_frame_.unsqueeze(0), int(down_sample_rate)).cuda()
                             for src_frame_ in src_frames.squeeze(0)]

            Rs_src = [pose[:3, :3] for pose in src_cam_poses.squeeze(0)]
            ts_src = [pose[:3, 3] for pose in src_cam_poses.squeeze(0)]

            # Warp the src-frames to the ref. view #
            if mGPU:
                WAPRED_src_frames = warp_homo.warp_img_feats_mgpu(
                    src_frames_dw, self.d_candi, Rs_src, ts_src, IntMs, unit_ray_Ms_2D)
            else:
                cam_intrin = cam_intrinsics[int(BatchIdx)]
                WAPRED_src_frames = warp_homo.warp_img_feats_v3(src_frames_dw, self.d_candi, Rs_src, ts_src,
                                                                cam_intrin, )

            ref_frame_dw_rep = torch.transpose(ref_frame_dw.repeat([len(self.d_candi), 1, 1, 1]), 0, 1)

            # Input to the KV-net #
            # torch.cat(tuple(WAPRED_src_frames), dim=0).shape: [3*4, D, H/4, W/4]
            # ref_frame_dw_rep.shape: [3, D, H/4, W/4]
            # (BV_cur - BV_predict).shape: [1, D, H/4, W/4]
            # kvnet_in_vol.shape: [1, 16, D, H/4, W/4]

            kvnet_in_vol = torch.cat(
                (torch.cat(tuple(WAPRED_src_frames), dim=0), ref_frame_dw_rep, DPV), dim=0).unsqueeze(0)

            # Run KV-net #
            BV_gain = self.kv_net(kvnet_in_vol)

            DPV = torch.squeeze(BV_gain, dim=1)
            DPV = F.log_softmax(DPV, dim=1)

            if self.if_refined:
                if self.refineNet_name == 'DPV':
                    dmap_cur_refined = self.r_net(torch.exp(BV_cur), img_features=d_net_features)  # [1, D, H, W]
                    dmap_refined = self.r_net(torch.exp(DPV), img_features=d_net_features)
                else:
                    raise NotImplementedError
            else:
                dmap_refined = -1
                dmap_cur_refined = -1

            return dmap_cur_refined, dmap_refined, BV_cur, DPV


# ver5-aggr_occ
class KVNET5(nn.Module):
    r'''
    Inside this module, we will do the full KV-Net pipeline:
    * D-Net (feature extractiion + BV_cur estimation )
    * KV-Net
    '''

    def __init__(self, feature_dim, cam_intrinsics, d_candi, sigma_soft_max,
                 KVNet_feature_dim, d_upsample_ratio_KV_net,
                 if_refined=True, refineNet_name='DPV',
                 t_win_r=2, refine_channel=3, if_upsample_d=False):
        r'''
        inputs:
        refineNet_name - {'DGF', 'DPV'}
        if_refined - if use the refinement net for upsampling
        refine_channel - the # of channels for the guided image in the refinement net, by default 3 for rgb input image
        '''

        super(KVNET5, self).__init__()
        self.t_win_r = t_win_r
        self.feature_dim = feature_dim
        self.KVNet_feature_dim = KVNet_feature_dim
        self.sigma_soft_max = sigma_soft_max
        self.d_upsample_ratio_KV_net = d_upsample_ratio_KV_net
        self.d_candi = d_candi
        self.if_refined = if_refined
        self.refineNet_name = refineNet_name
        self.if_upsample_d = if_upsample_d

        # submodules #
        if self.if_refined and self.refineNet_name == 'DPV':
            self.feature_extractor = model_basic.feature_extractor(feature_dim=feature_dim, multi_scale=True)


        else:
            self.feature_extractor = model_basic.feature_extractor(feature_dim=feature_dim)

        self.kv_net = model_basic.KV_NET_BASIC(3 * (t_win_r * 2 + 1) + 1,
                                               feature_dim=KVNet_feature_dim,
                                               up_sample_ratio=d_upsample_ratio_KV_net)

        if self.if_refined:
            if self.refineNet_name == 'DGF':
                self.r_net = model_refine.RefineNet_DGF(refine_channel)
            elif self.refineNet_name == 'DPV':
                self.r_net = model_refine.RefineNet_DPV_upsample(
                    int(self.feature_dim), int(self.feature_dim / 2), 3,
                    D=len(self.d_candi), upsample_D=self.if_upsample_d)
        self.transformer_3D = model_transformer.NONLocalBlock3D(KVNet_feature_dim, sub_sample=False, bn_layer=True,
                                                                ref=True)

        # print #
        print('KV-Net initialization:')
        print('with R-net: %r' % (self.if_refined))
        if self.if_refined:
            print('\trefinement name: %s' % (self.refineNet_name))

    def forward(self, ref_frame, src_frames, src_cam_poses, cam_intrinsics, BV_predict=None, IntMs=None,
                unit_ray_Ms_2D=None, mGPU=False, BatchIdx=None):
        r'''
        ver5-aggr_occ
        '''

        # D-Net #
        if (self.if_refined is False) or (self.if_refined is True and self.refineNet_name != 'DPV'):
            raise NotImplementedError

        else:
            # here
            feat_imgs_layer_1, feat_imgs = self.feature_extractor(torch.cat((src_frames[0, ...], ref_frame), dim=0))
            feat_img_ref_layer1 = feat_imgs_layer_1[-1, ...].unsqueeze(0)
            feat_img_ref = feat_imgs[-1, ...].unsqueeze(0)  # [1, D, H/4, W/4]
            feat_imgs_src = feat_imgs[:-1, ...].unsqueeze(0)  # [1, 4, D, H/4, W/4]
            d_net_features = [feat_img_ref, feat_img_ref_layer1, ref_frame]

        # warp src features to ref view
        warped_src_feats = []
        N = 0
        for src_num in range(feat_imgs_src.shape[1]):
            rel_Rt = src_cam_poses[N, src_num, :, :]  # .inverse()
            warped_src_feat = warp_homo.resample_vol_cuda(
                # src_vol=feat_imgs_src[:, src_num],
                src_vol=F.log_softmax(feat_imgs_src[:, src_num].clone(), dim=1),
                rel_extM=rel_Rt.cuda(feat_imgs_src.get_device()),
                cam_intrinsic=cam_intrinsics[BatchIdx[0].item()],
                d_candi=self.d_candi,
                padding_value=math.log(1. / float(len(self.d_candi)))
            ).clamp(max=0, min=-1000.).unsqueeze(0)
            warped_src_feats.append(warped_src_feat)
        warped_src_feats = torch.cat(warped_src_feats, dim=0)
        feat_imgs[:-1] = warped_src_feats
        feat_imgs = F.log_softmax(feat_imgs, dim=1)  # logSM version

        BV_cur = F.log_softmax(feat_img_ref, dim=1)

        # feat_imgs.shape: [V, D, H, W]
        # out.shape: [V, H, W]
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        feat_imgs1 = torch.exp(feat_imgs.detach())
        similarity = cos(feat_imgs1[-1].unsqueeze(0), feat_imgs1)
        occ_mask = (similarity > 0.5)

        feat_imgs_masked = torch.exp(feat_imgs) * occ_mask.unsqueeze(1).to(torch.float)
        # transformer! input: [b,c,t,h,w]
        DPV = self.transformer_3D.forward_ref(feat_imgs_masked.unsqueeze(0).permute(0, 2, 1, 3, 4), ref_idx=-1)

        if isinstance(BV_predict, torch.Tensor):
            raise Exception("BV_predict should be None!")

        else:
            # KV-Net #
            down_sample_rate = ref_frame.shape[3] / BV_cur.shape[3]

            ref_frame_dw = F.avg_pool2d(ref_frame, int(down_sample_rate)).cuda()
            src_frames_dw = [F.avg_pool2d(src_frame_.unsqueeze(0), int(down_sample_rate)).cuda()
                             for src_frame_ in src_frames.squeeze(0)]

            Rs_src = [pose[:3, :3] for pose in src_cam_poses.squeeze(0)]
            ts_src = [pose[:3, 3] for pose in src_cam_poses.squeeze(0)]

            # Warp the src-frames to the ref. view #
            if mGPU:
                WAPRED_src_frames = warp_homo.warp_img_feats_mgpu(
                    src_frames_dw, self.d_candi, Rs_src, ts_src, IntMs, unit_ray_Ms_2D)
            else:
                cam_intrin = cam_intrinsics[int(BatchIdx)]
                WAPRED_src_frames = warp_homo.warp_img_feats_v3(src_frames_dw, self.d_candi, Rs_src, ts_src,
                                                                cam_intrin, )

            ref_frame_dw_rep = torch.transpose(ref_frame_dw.repeat([len(self.d_candi), 1, 1, 1]), 0, 1)

            kvnet_in_vol = torch.cat(
                (torch.cat(tuple(WAPRED_src_frames), dim=0), ref_frame_dw_rep, DPV), dim=0).unsqueeze(0)

            # Run KV-net #
            BV_gain = self.kv_net(kvnet_in_vol)

            DPV = torch.squeeze(BV_gain, dim=1)
            DPV = F.log_softmax(DPV, dim=1)

            if self.if_refined:
                if self.refineNet_name == 'DPV':
                    dmap_cur_refined = self.r_net(torch.exp(BV_cur), img_features=d_net_features)  # [1, D, H, W]
                    dmap_refined = self.r_net(torch.exp(DPV), img_features=d_net_features)
                else:
                    raise NotImplementedError
            else:
                dmap_refined = -1
                dmap_cur_refined = -1

            return dmap_cur_refined, dmap_refined, BV_cur, DPV


# v6: aggregation + Knet Rnet together
class KVNET6(nn.Module):
    r'''
    Inside this module, we will do the full KV-Net pipeline:
    * D-Net (feature extractiion + BV_cur estimation )
    * KV-Net
    '''

    def __init__(self, feature_dim, cam_intrinsics, d_candi, sigma_soft_max,
                 KVNet_feature_dim, d_upsample_ratio_KV_net,
                 if_refined=True, refineNet_name='DPV',
                 t_win_r=2, refine_channel=3, if_upsample_d=False):
        r'''
        inputs:
        refineNet_name - {'DGF', 'DPV'}
        if_refined - if use the refinement net for upsampling
        refine_channel - the # of channels for the guided image in the refinement net, by default 3 for rgb input image
        '''

        super(KVNET6, self).__init__()
        self.t_win_r = t_win_r
        self.feature_dim = feature_dim
        self.KVNet_feature_dim = KVNet_feature_dim
        self.sigma_soft_max = sigma_soft_max
        self.d_upsample_ratio_KV_net = d_upsample_ratio_KV_net
        self.d_candi = d_candi
        self.if_refined = if_refined
        self.refineNet_name = refineNet_name
        self.if_upsample_d = if_upsample_d

        # submodules #
        if self.if_refined and self.refineNet_name == 'DPV':
            self.feature_extractor = model_basic.feature_extractor(feature_dim=feature_dim, multi_scale=True)


        else:
            self.feature_extractor = model_basic.feature_extractor(feature_dim=feature_dim)
        #
        # self.kv_net = model_basic.KV_NET_BASIC(3 * (t_win_r * 2 + 1) + 1,
        #                                        feature_dim=KVNet_feature_dim,
        #                                        up_sample_ratio=d_upsample_ratio_KV_net)

        if self.if_refined:
            if self.refineNet_name == 'DGF':
                self.kr_net = model_refine.RefineNet_DGF(refine_channel)
            elif self.refineNet_name == 'DPV':
                self.kr_net = model_refine.Regularize_and_RefineNet_DPV_upsample(
                    3 * (t_win_r * 2 + 1) + 1,
                    feature_dim=KVNet_feature_dim,
                    up_sample_ratio=d_upsample_ratio_KV_net,
                    C0=int(self.feature_dim), C1=int(self.feature_dim / 2), C2=3,
                    D=len(self.d_candi), upsample_D=self.if_upsample_d)
        self.transformer_3D = model_transformer.NONLocalBlock3D(KVNet_feature_dim, sub_sample=False, bn_layer=True,
                                                                ref=True)

        # print #
        print('KV-Net initialization:')
        print('with R-net: %r' % (self.if_refined))
        if self.if_refined:
            print('\trefinement name: %s' % (self.refineNet_name))

    def forward(self, ref_frame, src_frames, src_cam_poses, cam_intrinsics, BV_predict=None, IntMs=None,
                unit_ray_Ms_2D=None, mGPU=False, BatchIdx=None):

        r'''
        # v6: aggregation + Knet Rnet together
        '''

        # D-Net #
        if (self.if_refined is False) or (self.if_refined is True and self.refineNet_name != 'DPV'):
            raise NotImplementedError

        else:
            # here
            feat_imgs_layer_1, feat_imgs = self.feature_extractor(torch.cat((src_frames[0, ...], ref_frame), dim=0))
            feat_img_ref_layer1 = feat_imgs_layer_1[-1, ...].unsqueeze(0)
            feat_img_ref = feat_imgs[-1, ...].unsqueeze(0)  # [1, D, H/4, W/4]
            feat_imgs_src = feat_imgs[:-1, ...].unsqueeze(0)  # [1, 4, D, H/4, W/4]
            d_net_features = [feat_img_ref, feat_img_ref_layer1, ref_frame]

        # warp src features to ref view
        warped_src_feats = []
        N = 0
        for src_num in range(feat_imgs_src.shape[1]):
            rel_Rt = src_cam_poses[N, src_num, :, :]  # .inverse()
            warped_src_feat = warp_homo.resample_vol_cuda(
                # src_vol=feat_imgs_src[:, src_num],
                src_vol=F.log_softmax(feat_imgs_src[:, src_num].clone(), dim=1),
                rel_extM=rel_Rt.cuda(feat_imgs_src.get_device()),
                cam_intrinsic=cam_intrinsics[BatchIdx[0].item()],
                d_candi=self.d_candi,
                padding_value=math.log(1. / float(len(self.d_candi)))
            ).clamp(max=0, min=-1000.).unsqueeze(0)
            warped_src_feats.append(warped_src_feat)
        warped_src_feats = torch.cat(warped_src_feats, dim=0)
        feat_imgs[:-1] = warped_src_feats
        feat_imgs = F.log_softmax(feat_imgs, dim=1)  # logSM version

        BV_cur = F.log_softmax(feat_img_ref, dim=1)

        DPV = self.transformer_3D.forward_ref(torch.exp(feat_imgs).unsqueeze(0).permute(0, 2, 1, 3, 4), ref_idx=-1)

        if isinstance(BV_predict, torch.Tensor):
            raise Exception("BV_predict should be None!")

        else:
            # KV-Net #
            down_sample_rate = ref_frame.shape[3] / BV_cur.shape[3]

            ref_frame_dw = F.avg_pool2d(ref_frame, int(down_sample_rate)).cuda()
            src_frames_dw = [F.avg_pool2d(src_frame_.unsqueeze(0), int(down_sample_rate)).cuda()
                             for src_frame_ in src_frames.squeeze(0)]

            Rs_src = [pose[:3, :3] for pose in src_cam_poses.squeeze(0)]
            ts_src = [pose[:3, 3] for pose in src_cam_poses.squeeze(0)]

            # Warp the src-frames to the ref. view #
            if mGPU:
                WAPRED_src_frames = warp_homo.warp_img_feats_mgpu(
                    src_frames_dw, self.d_candi, Rs_src, ts_src, IntMs, unit_ray_Ms_2D)
            else:
                cam_intrin = cam_intrinsics[int(BatchIdx)]
                WAPRED_src_frames = warp_homo.warp_img_feats_v3(src_frames_dw, self.d_candi, Rs_src, ts_src,
                                                                cam_intrin, )

            ref_frame_dw_rep = torch.transpose(ref_frame_dw.repeat([len(self.d_candi), 1, 1, 1]), 0, 1)

            # Input to the KV-net #
            # torch.cat(tuple(WAPRED_src_frames), dim=0).shape: [3*4, D, H/4, W/4]
            # ref_frame_dw_rep.shape: [3, D, H/4, W/4]
            # (BV_cur - BV_predict).shape: [1, D, H/4, W/4]
            # kvnet_in_vol.shape: [1, 16, D, H/4, W/4]
            # ref_frame_dw_rep.shape: [3, D, H/4, W/4]

            # Run KV-net #
            # BV_gain = self.kv_net(kvnet_in_vol)
            # DPV = torch.squeeze(BV_gain, dim=1)
            DPV = F.log_softmax(DPV, dim=1)

            kvnet_in_vol_all = torch.cat(
                (torch.cat(tuple(WAPRED_src_frames), dim=0), ref_frame_dw_rep, DPV), dim=0).unsqueeze(0)
            dmap_refined = self.kr_net(kvnet_in_vol_all, img_features=d_net_features)

            cc, dd, hh, ww = ref_frame_dw_rep.shape
            kvnet_in_vol_ref = torch.cat(
                (ref_frame_dw_rep.expand(5, cc, dd, hh, ww).reshape(5 * cc, dd, hh, ww), BV_cur), dim=0).unsqueeze(0)
            dmap_cur_refined = self.kr_net(kvnet_in_vol_ref, img_features=d_net_features)

            return dmap_cur_refined, dmap_refined, BV_cur, DPV


# v7: aggregation + rnet only
class KVNET7(nn.Module):
    r'''
    Inside this module, we will do the full KV-Net pipeline:
    * D-Net (feature extractiion + BV_cur estimation )
    * KV-Net
    '''

    def __init__(self, feature_dim, cam_intrinsics, d_candi, sigma_soft_max,
                 KVNet_feature_dim, d_upsample_ratio_KV_net,
                 if_refined=True, refineNet_name='DPV',
                 t_win_r=2, refine_channel=3, if_upsample_d=False):
        r'''
        inputs:
        refineNet_name - {'DGF', 'DPV'}
        if_refined - if use the refinement net for upsampling
        refine_channel - the # of channels for the guided image in the refinement net, by default 3 for rgb input image
        '''

        super(KVNET7, self).__init__()
        self.t_win_r = t_win_r
        self.feature_dim = feature_dim
        self.KVNet_feature_dim = KVNet_feature_dim
        self.sigma_soft_max = sigma_soft_max
        self.d_upsample_ratio_KV_net = d_upsample_ratio_KV_net
        self.d_candi = d_candi
        self.if_refined = if_refined
        self.refineNet_name = refineNet_name
        self.if_upsample_d = if_upsample_d

        # submodules #
        if self.if_refined and self.refineNet_name == 'DPV':
            self.feature_extractor = model_basic.feature_extractor(feature_dim=feature_dim, multi_scale=True)


        else:
            self.feature_extractor = model_basic.feature_extractor(feature_dim=feature_dim)

        # self.kv_net = model_basic.KV_NET_BASIC(3 * (t_win_r * 2 + 1) + 1,
        #                                        feature_dim=KVNet_feature_dim,
        #                                        up_sample_ratio=d_upsample_ratio_KV_net)

        if self.if_refined:
            if self.refineNet_name == 'DGF':
                self.r_net = model_refine.RefineNet_DGF(refine_channel)
            elif self.refineNet_name == 'DPV':
                self.r_net = model_refine.RefineNet_DPV_upsample(
                    int(self.feature_dim), int(self.feature_dim / 2), 3,
                    D=len(self.d_candi), upsample_D=self.if_upsample_d)
        self.transformer_3D = model_transformer.NONLocalBlock3D(KVNet_feature_dim, sub_sample=False, bn_layer=True,
                                                                ref=True)

        # print #
        print('KV-Net initialization:')
        print('with R-net: %r' % (self.if_refined))
        if self.if_refined:
            print('\trefinement name: %s' % (self.refineNet_name))

    def forward(self, ref_frame, src_frames, src_cam_poses, cam_intrinsics, BV_predict=None, IntMs=None,
                unit_ray_Ms_2D=None, mGPU=False, BatchIdx=None):
        r'''
        # v7: aggregation + rnet only

        '''

        # D-Net #
        if (self.if_refined is False) or (self.if_refined is True and self.refineNet_name != 'DPV'):
            raise NotImplementedError

        else:
            # here
            feat_imgs_layer_1, feat_imgs = self.feature_extractor(torch.cat((src_frames[0, ...], ref_frame), dim=0))
            feat_img_ref_layer1 = feat_imgs_layer_1[-1, ...].unsqueeze(0)
            feat_img_ref = feat_imgs[-1, ...].unsqueeze(0)  # [1, D, H/4, W/4]
            feat_imgs_src = feat_imgs[:-1, ...].unsqueeze(0)  # [1, 4, D, H/4, W/4]
            d_net_features = [feat_img_ref, feat_img_ref_layer1, ref_frame]

        # warp src features to ref view
        warped_src_feats = []
        N = 0
        for src_num in range(feat_imgs_src.shape[1]):
            rel_Rt = src_cam_poses[N, src_num, :, :]  # .inverse()
            warped_src_feat = warp_homo.resample_vol_cuda(
                # src_vol=feat_imgs_src[:, src_num],
                src_vol=F.log_softmax(feat_imgs_src[:, src_num].clone(), dim=1),
                rel_extM=rel_Rt.cuda(feat_imgs_src.get_device()),
                cam_intrinsic=cam_intrinsics[BatchIdx[0].item()],
                d_candi=self.d_candi,
                padding_value=math.log(1. / float(len(self.d_candi)))
            ).clamp(max=0, min=-1000.).unsqueeze(0)
            warped_src_feats.append(warped_src_feat)
        warped_src_feats = torch.cat(warped_src_feats, dim=0)
        feat_imgs[:-1] = warped_src_feats
        feat_imgs = F.log_softmax(feat_imgs, dim=1)  # logSM version

        BV_cur = F.log_softmax(feat_img_ref, dim=1)

        DPV = self.transformer_3D.forward_ref(torch.exp(feat_imgs).unsqueeze(0).permute(0, 2, 1, 3, 4), ref_idx=-1)

        if isinstance(BV_predict, torch.Tensor):
            raise Exception("BV_predict should be None!")

        else:
            DPV = F.log_softmax(DPV, dim=1)

            if self.if_refined:
                if self.refineNet_name == 'DPV':
                    dmap_cur_refined = self.r_net(torch.exp(BV_cur), img_features=d_net_features)  # [1, D, H, W]
                    dmap_refined = self.r_net(torch.exp(DPV), img_features=d_net_features)
                else:
                    raise NotImplementedError
            else:
                dmap_refined = -1
                dmap_cur_refined = -1

            return dmap_cur_refined, dmap_refined, BV_cur, DPV
