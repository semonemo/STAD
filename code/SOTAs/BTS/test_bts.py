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

import matplotlib.pyplot as plt
from bts_networks.bts_enc_dec import *
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


def get_focal(img_path):
    focal = 721.5377  # 522
    scene_name = img_path.split('/')[5]
    if scene_name == '2011_09_28_drive_0002_sync':
        focal = 707.0493  # 25
    elif scene_name == '2011_09_29_drive_0071_sync':
        focal = 718.3351  # 25
    elif scene_name == '2011_09_30_drive_0016_sync':
        focal = 707.0912  # 75
    elif scene_name == '2011_10_03_drive_0027_sync':
        focal = 718.856  # 50

    return focal


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
    parser.add_argument('--d_max', type=float, default=0, help='The minimal depth value; default=0')
    parser.add_argument('--max_depth', type=float, default=5, help='The maximal depth value; default=15')  ##
    parser.add_argument('--ndepth', type=int, default=64, help='The # of candidate depth values; default= 128')
    parser.add_argument('--sigma_soft_max', type=float, default=10., help='sigma_soft_max, default = 500.')
    parser.add_argument('--feature_dim', type=int, default=64,
                        help='The feature dimension for the feature extractor; default=64')

    # about dataset #
    parser.add_argument('--dataset', type=str, default='scanNet', help='Dataset name: {scanNet, 7scenes, kitti}')
    parser.add_argument('--dataset_path', type=str, default='.', help='Path to the dataset')
    parser.add_argument('--change_aspect_ratio', action='store_true', default=False,
                        help='If we want to change the aspect ratio. This option is only useful for KITTI')

    parser.add_argument('--encoder', type=str, default='densenet121_bts')

    # parsing parameters #
    args = parser.parse_args()
    args.max_depth = args.d_max
    d_candi = np.linspace(args.d_min, args.d_max, args.ndepth)
    d_upsample = None
    d_candi_dmap_ref = d_candi

    split_file = args.split_file

    # ===== Dataset selection ======== #
    dataset_path = args.dataset_path
    if args.dataset == 'scanNet':
        import mdataloader.scanNet as dl_scanNet
        dataset_init = dl_scanNet.ScanNet_dataset
        fun_get_paths = lambda traj_indx: dl_scanNet.get_paths(traj_indx, frame_interv=args.frame_interv,
                                                               split_txt=split_file,
                                                               database_path_base=dataset_path)
        img_size = [384, 256]

        # trajectory index for testing #
        n_scenes, _, _, _, _ = fun_get_paths(0)
        traj_Indx = np.arange(0, n_scenes)

    elif args.dataset == '7scenes':
        # 7 scenes video #
        import mdataloader.dl_7scenes as dl_7scenes
        dataset_init = dl_7scenes.SevenScenesDataset
        dat_indx_step = 3

        split_file = None if args.split_file == '.' else args.split_file
        fun_get_paths = lambda traj_indx: dl_7scenes.get_paths_1frame(
            traj_indx, database_path_base=dataset_path, split_txt=split_file,
            dat_indx_step=dat_indx_step)

        img_size = [384, 256]
        n_scenes, _, _, _, _ = fun_get_paths(0)
        traj_Indx = np.arange(0, n_scenes)

    elif args.dataset == 'kitti':
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

        # img_size = [1024, 320]  ###
        n_scenes, _, _, _, _ = fun_get_paths(0)
        traj_Indx = np.arange(0, n_scenes)

    else:
        raise Exception('dataset loader not implemented')

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

    print('Initnializing BTS')
    exp_name = 'BTS'
    wandb.init(project="stad", name=f'Eval/{exp_name}-{len(traj_Indx)}', config=args)

    bts_weights_folder = '/data/checkpoints/bts_eigen_v2_pytorch_densenet121/model'
    args.encoder = bts_weights_folder.split('/')[-2].split('_')[-1] + '_bts'
    model = BtsModel(args)
    model = torch.nn.DataParallel(model)
    model.cuda()
    total_params = count_parameters(model)

    # load pretrained model
    print("-> Loading weights from {}".format(bts_weights_folder))
    checkpoint = torch.load(bts_weights_folder)
    model.load_state_dict(checkpoint['model'])
    print("Loaded checkpoint '{}' (global_step {})".format(bts_weights_folder, checkpoint['global_step']))

    model.eval()

    pred_depths = []
    gt_depths = []
    eval_measures = torch.zeros(10).cuda()

    #### eigen only
    with open(args.split_eigen) as f:
        lines = f.readlines()
    eigen_split = [args.dataset_path + '/rawdata/' + line.split(' ')[0] for line in lines]
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
            # #### eigen only
            # if ref_dat['img_path'] not in eigen_split:
            #     dat_array.pop(0)
            #     dat_array.append(dataset[ref_indx + args.t_win + 1])
            #     continue

            print(ref_dat['img_path'])
            if valid_seq and eff_iter:
                print('testing on %d/%d frame in traj %d/%d ... ' % \
                      (frame_cnt + 1, traj_length - 2 * args.t_win, traj_idx + 1, len(traj_Indx)))
                # pass image to monodepth2
                input_color = ref_dat['img'].cuda()
                focal = get_focal(ref_dat['img_path'])
                focal = torch.autograd.Variable(torch.tensor([focal])).cuda()
                lpg8x8, lpg4x4, lpg2x2, reduc1x1, depth_est = model(input_color, focal)

                pred_depth = depth_est.cpu().detach().numpy().squeeze()
                # pred_depth = depth_est.cpu()[:, 0].detach().numpy().squeeze()

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
                save_folder = '/data/out/kitti/SOTA_BTS/'
                save_folder_png = '/data/out/kitti/SOTA_BTS_png'
                save_folder_png_error = '/data/out/kitti/SOTA_BTS_png/error'
                save_folder_png_disp = '/data/out/kitti/SOTA_BTS_png/disp'

                scene_name = img_name.split('/')[-4]
                img_number = img_name.split('/')[-1][:-4]
                os.makedirs(save_folder + '/' + scene_name, exist_ok=True)
                os.makedirs(save_folder_png_error + '/' + scene_name, exist_ok=True)
                os.makedirs(save_folder_png_disp + '/' + scene_name, exist_ok=True)
                depth_save_name = save_folder + '/' + scene_name + '/depth_' + img_number + '.npy'
                error_save_name = save_folder_png_error + '/' + scene_name + '/error_' + img_number + '.png'
                png_disparity_save_name = save_folder_png_disp + '/' + scene_name + '/disp_' + img_number + '.png'

                rgb_save_name = save_folder + '/' + scene_name + '/rgb_' + img_number + '.png'
                gt_depth_save_name = save_folder + '/' + scene_name + '/gt_' + img_number + '.npy'
                intrin_save_name = save_folder + '/' + scene_name + '/intrinsic.npy'

                unnorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                plt.imsave(rgb_save_name,
                           unnorm(dataset[ref_indx]['img'][0]).permute(1, 2, 0).cpu().detach().numpy())

                plt.imsave(png_disparity_save_name, 1 / pred_depth, cmap='inferno')

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
