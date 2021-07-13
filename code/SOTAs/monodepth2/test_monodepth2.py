'''
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license

author: Chao Liu <chaoliu1@cs.cmu.edu>
'''
# test #
import numpy as np
import torch
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

torch.backends.cudnn.benchmark = True

import warping.homography as warp_homo
import mutils.misc as m_misc
# import models.KVNET as m_kvnet
# import models.KVNET_transformer as m_kvnet

import utils.models as utils_model
import test_utils.export_res as export_res
import test_utils.test_KVNet as test_KVNet

import matplotlib as mlt
import wandb
import mono2_networks
from layers import disp_to_depth
import matplotlib.pyplot as plt
from utils.misc import count_parameters
from mdataloader.m_preprocess import UnNormalize

mlt.use('Agg')


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
        if np.isnan(dat['extM'].min()) or np.isnan(dat['extM'].max()):
            if_valid = False
            break

        elif isinstance(dat['extM'], int):
            if_valid = False
            break

    return if_valid


def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    d1 = (thresh < 1.25).mean()
    d2 = (thresh < 1.25 ** 2).mean()
    d3 = (thresh < 1.25 ** 3).mean()

    rms = (gt - pred) ** 2
    rms = np.sqrt(rms.mean())

    log_rms = (np.log(gt) - np.log(pred)) ** 2
    log_rms = np.sqrt(log_rms.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    err = np.abs(np.log10(pred) - np.log10(gt))
    log10 = np.mean(err)

    return [silog, abs_rel, log10, rms, sq_rel, log_rms, d1, d2, d3]


def main():
    import argparse
    print('Parsing the arguments...')
    parser = argparse.ArgumentParser()

    # exp name #
    parser.add_argument('--exp_name', required=True, type=str,
                        help='The name of the experiment. Used to naming the folders')

    # about testing #
    parser.add_argument('--split_file', type=str, required=True, help='The split txt file')
    parser.add_argument('--split_eigen', type=str, default='./mdataloader/kitti_split/test_eigen.txt',
                        help='The split eigen txt file')
    parser.add_argument('--frame_interv', default=5, type=int, help='frame interval')
    parser.add_argument('--t_win', type=int, default=2, help='The radius of the temporal window; default=2')
    parser.add_argument('--d_min', type=float, default=0, help='The minimal depth value; default=0')
    parser.add_argument('--d_max', type=float, default=5, help='The maximal depth value; default=15')
    parser.add_argument('--ndepth', type=int, default=64, help='The # of candidate depth values; default= 128')
    parser.add_argument('--sigma_soft_max', type=float, default=10., help='sigma_soft_max, default = 500.')
    parser.add_argument('--feature_dim', type=int, default=64,
                        help='The feature dimension for the feature extractor; default=64')

    # about dataset #
    parser.add_argument('--dataset', type=str, default='scanNet', help='Dataset name: {scanNet, 7scenes, kitti}')
    parser.add_argument('--dataset_path', type=str, default='.', help='Path to the dataset')
    parser.add_argument('--change_aspect_ratio', action='store_true', default=False,
                        help='If we want to change the aspect ratio. This option is only useful for KITTI')

    # parsing parameters #
    args = parser.parse_args()
    d_candi = np.linspace(args.d_min, args.d_max, args.ndepth)
    d_upsample = None
    d_candi_dmap_ref = d_candi

    split_file = args.split_file

    # ===== Dataset selection ======== #
    dataset_path = args.dataset_path
    if args.dataset == 'kitti':
        import mdataloader.kitti as dl_kitti
        dataset_init = dl_kitti.KITTI_dataset
        if not dataset_path == '.':
            fun_get_paths = lambda traj_indx: dl_kitti.get_paths(traj_indx, split_txt=split_file,
                                                                 mode='val', database_path_base=dataset_path)
        else:  # use default database path
            fun_get_paths = lambda traj_indx: dl_kitti.get_paths(traj_indx, split_txt=split_file, mode='val')
        if not args.change_aspect_ratio:  # we will keep the aspect ratio and do cropping
            img_size = [768, 256]
            crop_w = None
        else:  # we will change the aspect ratio and NOT do cropping
            img_size = [384, 256]
            crop_w = None

        # img_size = [1024,320] ###
        n_scenes, _, _, _, _ = fun_get_paths(0)
        traj_Indx = np.arange(0, n_scenes)

    else:
        raise Exception('dataset loader not implemented')

    fldr_path, img_paths, dmap_paths, poses, intrin_path = fun_get_paths(0)
    if args.dataset == 'kitti':
        dataset = dataset_init(True, img_paths, dmap_paths, poses,
                               intrin_path=intrin_path, img_size=img_size, digitize=True,
                               d_candi=d_candi_dmap_ref, resize_dmap=.25, crop_w=crop_w)

        dataset_imgsize = dataset_init(True, img_paths, dmap_paths, poses,
                                       intrin_path=intrin_path, img_size=img_size, digitize=True,
                                       d_candi=d_candi_dmap_ref, resize_dmap=1)
    else:
        dataset = dataset_init(True, img_paths, dmap_paths, poses,
                               intrin_path=intrin_path, img_size=img_size, digitize=True,
                               d_candi=d_candi_dmap_ref, resize_dmap=.25)

        dataset_imgsize = dataset_init(True, img_paths, dmap_paths, poses,
                                       intrin_path=intrin_path, img_size=img_size, digitize=True,
                                       d_candi=d_candi_dmap_ref, resize_dmap=1)
    # ================================ #

    print('Initnializing MonoDepth2')
    encoder = mono2_networks.ResnetEncoder(18, False)
    depth_decoder = mono2_networks.DepthDecoder(encoder.num_ch_enc)

    # load pretrained model
    mono2_weights_folder = '/data/checkpoints/monodepth2/mono_1024x320/'
    print("-> Loading weights from {}".format(mono2_weights_folder))
    encoder_path = os.path.join(mono2_weights_folder, "encoder.pth")
    decoder_path = os.path.join(mono2_weights_folder, "depth.pth")

    encoder_dict = torch.load(encoder_path)
    feed_size = [encoder_dict['height'], encoder_dict['width']]

    model_dict = encoder.state_dict()
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
    depth_decoder.load_state_dict(torch.load(decoder_path))

    encoder = torch.nn.DataParallel(encoder)
    depth_decoder = torch.nn.DataParallel(depth_decoder)
    encoder.cuda()
    depth_decoder.cuda()

    enc_params = count_parameters(encoder)
    dec_params = count_parameters(depth_decoder)
    total_params = enc_params + dec_params

    encoder.eval()
    depth_decoder.eval()

    # model_KVnet = m_kvnet.KVNET(feature_dim=args.feature_dim, cam_intrinsics=dataset.cam_intrinsics,
    #                             d_candi=d_candi, sigma_soft_max=args.sigma_soft_max, KVNet_feature_dim=args.feature_dim,
    #                             d_upsample_ratio_KV_net=d_upsample, t_win_r=args.t_win, if_refined=True)
    #
    # model_KVnet = torch.nn.DataParallel(model_KVnet)
    # model_KVnet.cuda()

    exp_name = 'MonoDepth2'
    wandb.init(project="stad", name=f'Eval/{exp_name}-{len(traj_Indx)}', config=args)

    pred_depths = []
    gt_depths = []
    eval_measures = torch.zeros(10).cuda()

    for traj_idx in traj_Indx:
        res_fldr = '../results/%s/traj_%d' % (args.exp_name, traj_idx)
        m_misc.m_makedir(res_fldr)
        scene_path_info = []

        print(f'Getting the paths for traj ({traj_idx}/{len(traj_Indx)}')
        fldr_path, img_seq_paths, dmap_seq_paths, poses, intrin_path = fun_get_paths(traj_idx)
        dataset.set_paths(img_seq_paths, dmap_seq_paths, poses)

        if args.dataset is 'scanNet':
            # For each trajector in the dataset, we will update the intrinsic matrix #
            dataset.get_cam_intrinsics(intrin_path)

        print('Done')
        dat_array = [dataset[idx] for idx in range(args.t_win * 2 + 1)]
        traj_length = len(dataset)
        print('trajectory length = %d' % (traj_length))

        for frame_cnt, ref_indx in enumerate(range(args.t_win, traj_length - args.t_win - 1)):
            eff_iter = True
            valid_seq = check_datArray_pose(dat_array)

            # Read ref. and src. data in the local time window #
            ref_dat, src_dats = m_misc.split_frame_list(dat_array, args.t_win)

            print(ref_dat['img_path'])
            if valid_seq and eff_iter:
                print('testing on %d/%d frame in traj %d/%d ... ' % \
                      (frame_cnt + 1, traj_length - 2 * args.t_win, traj_idx + 1, len(traj_Indx)))

                # pass image to monodepth2
                input_color = ref_dat['img'].cuda()
                input_color = torch.nn.functional.interpolate(input_color, size=feed_size, mode='bilinear',
                                                              align_corners=False)  ##

                output = depth_decoder(encoder(input_color))
                scaled_disp, pred_depth = disp_to_depth(output[("disp", 0)], args.d_min, args.d_max)

                pred_depth = torch.nn.functional.interpolate(pred_depth, size=ref_dat['img'].shape[-2:],
                                                             mode='bilinear', align_corners=False)  ##
                pred_depth = pred_depth.cpu()[:, 0].detach().numpy().squeeze()

                try:
                    gt_depth = ref_dat['dmap_imgsize'].squeeze().cpu().detach().numpy()
                except:
                    print("invalid gt, continue")
                    dat_array.pop(0)
                    dat_array.append(dataset[ref_indx + args.t_win + 1])
                    continue

                # median scaling
                # mask = gt_depth > 0
                mask = np.logical_and(gt_depth >= args.d_min, gt_depth <= args.d_max)

                ratio = np.median(gt_depth[mask]) / np.median(pred_depth[mask])
                pred_depth *= ratio

                measures = compute_errors(gt_depth[mask], pred_depth[mask])
                eval_measures[:9] += torch.FloatTensor(measures).cuda()
                eval_measures[9] += 1

                ########################################
                # saving rgb, depth to estimate Es, Ed
                img_name = dataset[ref_indx]['img_path']
                save_folder = '/data/out/kitti/predicted/SOTA_monodepth/'
                save_folder_png = '/data/out/kitti/predicted/SOTA_monodepth_png'
                save_folder_png_error = f'{save_folder_png}/error'
                save_folder_png_disp = f'{save_folder_png}/disp'
                save_folder_png_depth = f'{save_folder_png}/depth'
                save_folder_png_rgb = f'{save_folder_png}/rgb'
                save_folder_png_gt_disp = f'{save_folder_png}/gt_disp'

                scene_name = img_name.split('/')[-4]
                img_number = img_name.split('/')[-1][:-4]
                os.makedirs(save_folder + '/' + scene_name, exist_ok=True)
                os.makedirs(save_folder_png_error + '/' + scene_name, exist_ok=True)
                os.makedirs(save_folder_png_disp + '/' + scene_name, exist_ok=True)
                # os.makedirs(save_folder_png_depth + '/' + scene_name, exist_ok=True)
                os.makedirs(save_folder_png_rgb + '/' + scene_name, exist_ok=True)
                os.makedirs(save_folder_png_gt_disp + '/' + scene_name, exist_ok=True)

                error_save_name = save_folder_png_error + '/' + scene_name + '/error_' + img_number + '.png'
                png_disparity_save_name = save_folder_png_disp + '/' + scene_name + '/disp_' + img_number + '.png'
                # png_depth_save_name = save_folder_png_depth + '/' + scene_name + '/depth_' + img_number + '.png'
                rgb_save_name = save_folder_png_rgb + '/' + scene_name + '/rgb_' + img_number + '.png'
                png_gt_save_name = save_folder_png_gt_disp + '/' + scene_name + '/gt_' + img_number + '.png'

                depth_save_name = save_folder + '/' + scene_name + '/depth_' + img_number + '.npy'
                gt_depth_save_name = save_folder + '/' + scene_name + '/gt_' + img_number + '.npy'
                intrin_save_name = save_folder + '/' + scene_name + '/intrinsic.npy'

                unnorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                plt.imsave(rgb_save_name,
                           unnorm(dataset[ref_indx]['img'][0]).permute(1, 2, 0).cpu().detach().numpy())

                pred_depth[pred_depth < args.d_min] = args.d_min
                pred_depth[pred_depth > args.d_max] = args.d_max

                gt_mask = (gt_depth == 0)
                gt_disp = 1 / gt_depth
                gt_disp[gt_mask] = 0

                plt.imsave(png_disparity_save_name, 1 / pred_depth, cmap='inferno')
                # plt.imsave(png_depth_save_name, pred_depth, cmap='inferno')
                plt.imsave(png_gt_save_name, gt_disp, cmap='inferno')

                np.save(intrin_save_name, dataset.cam_intrinsics['intrinsic_M_cuda'].cpu().detach().numpy())
                np.save(depth_save_name, pred_depth)
                np.save(gt_depth_save_name, gt_depth)

                thresh = np.zeros_like(gt_depth)
                thresh[mask] = np.maximum((gt_depth[mask] / pred_depth[mask]), (pred_depth[mask] / gt_depth[mask]))
                d1 = (thresh < 1.25)
                plt.imsave(error_save_name, ~d1, cmap='gray')

            # Update dat_array #
            dat_array.pop(0)
            dat_array.append(dataset[ref_indx + args.t_win + 1])
        m_misc.save_ScenePathInfo('%s/scene_path_info.txt' % (res_fldr), scene_path_info)

    print("num pred_depths = ", len(pred_depths))
    eval_measures_cpu = eval_measures.cpu()
    cnt = eval_measures_cpu[9].item()
    eval_measures_cpu /= cnt
    print('Computing errors for {} eval samples'.format(int(cnt)))
    print("{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}"
          .format('silog', 'abs_rel', 'log10', 'rms', 'sq_rel', 'log_rms', 'd1', 'd2', 'd3'))
    for i in range(8):
        print('{:7.3f}, '.format(eval_measures_cpu[i]), end='')
    print('{:7.3f}'.format(eval_measures_cpu[8]))

    silog, abs_rel, log10, rms, sq_rel, log_rms, d1, d2, d3 = eval_measures_cpu[:9]
    wandb.log({'error/silog': silog,
               'error/log10': log10,
               'error/abs_rel': abs_rel,
               'error/sq_rel': sq_rel,
               'error/rms': rms,
               'error/log_rms': log_rms,
               'error/d1': d1,
               'error/d2': d2,
               'error/d3': d3,
               'tot_params': total_params}, step=0)

    wandb.log({'error/silog': silog,
               'error/log10': log10,
               'error/abs_rel': abs_rel,
               'error/sq_rel': sq_rel,
               'error/rms': rms,
               'error/log_rms': log_rms,
               'error/d1': d1,
               'error/d2': d2,
               'error/d3': d3,
               'tot_params': total_params}, step=100000)


if __name__ == '__main__':
    main()
