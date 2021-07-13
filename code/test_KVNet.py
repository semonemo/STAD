'''
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license

author: Chao Liu <chaoliu1@cs.cmu.edu>
'''
# test #
import numpy as np
import torch

torch.backends.cudnn.benchmark = True

import mutils.misc as m_misc
import warping.homography as warp_homo
from models.KVNET import KVNET as orig_kvnet
import models.KVNET_transformer as m_kvnet

import utils.models as utils_model
import test_utils.export_res as export_res
import test_utils.test_KVNet as test_KVNet

import matplotlib as mlt
import wandb
import os
from utils.misc import count_parameters
import matplotlib.pyplot as plt

mlt.use('Agg')

from mdataloader.m_preprocess import UnNormalize


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
    parser.add_argument('--exp_name', default=None, type=str,
                        help='The name of the experiment. Used to naming the folders')

    # about testing #
    parser.add_argument('--model_path', type=str, required=True, help='The pre-trained model path for KV-net')
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
        raise Exception('dataset loader not implemented')

    # try:
    #     global_step = int(os.path.basename(args.model_path).split('.')[0])
    #     model_dir = os.path.dirname(args.model_path)
    #     # global_steps = range(60000, global_step + 1, 2000) # opt1: test multiple steps
    #     global_steps = range(global_step, global_step+1, 2000) # opt2: test one step
    #     model_paths = [f'{model_dir}/{gs}.tar' for gs in global_steps]
    #     exp_name = args.model_path.split('/')[-3]
    #     try:
    #         version_num = int(exp_name.split('-')[0].split('ver')[-1])
    #     except:
    #         version_num = -1
    #
    # except:
    #     global_step = 0
    #     model_paths = [args.model_path]
    #     exp_name = 'git_pretraind'
    #     version_num = -1

    ###
    global_step = 0
    model_paths = [args.model_path]
    exp_name = args.model_path.split('/')[-1].split('.')[0]
    version_num = int(exp_name.split('-')[0].split('ver')[-1])

    save_folder_temp = os.path.dirname(os.path.dirname(args.model_path))
    save_folder_init = f'{os.path.dirname(save_folder_temp)}/predicted/{os.path.basename(save_folder_temp)}'


    print('Initnializing the KV-Net')
    if version_num == -1:
        print("**test with pretrained nrgbd method")
        model_version = orig_kvnet
        test_KVNet_test = test_KVNet.test
        save_folder_init = '/data/out/kitti/predicted/nrgbd_pretraind'

    elif version_num == 0:
        print("**test with original nrgbd(V0) method")
        model_version = orig_kvnet
        test_KVNet_test = test_KVNet.test

    elif version_num == 1:
        print("**test with per-frame(V1) method")
        model_version = m_kvnet.KVNET1
        test_KVNet_test = test_KVNet.test_no_pred_DPV

    elif version_num == 2:
        print("**test with SA with pred DPV (V2) method")
        model_version = m_kvnet.KVNET2
        test_KVNet_test = test_KVNet.test

    elif version_num == 3:
        print("**test with SA without pred DPV (V3) method")
        model_version = m_kvnet.KVNET3
        test_KVNet_test = test_KVNet.test_no_pred_DPV

    elif version_num == 4:
        print("**test with aggregation(V4) method")
        model_version = m_kvnet.KVNET4
        test_KVNet_test = test_KVNet.test_no_pred_DPV

    elif version_num == 5:
        print("**test with aggregation + occ mask (V5) method")
        model_version = m_kvnet.KVNET5
        test_KVNet_test = test_KVNet.test_no_pred_DPV

    elif version_num == 6:
        print("**test with aggregate+KR(V6) method")
        model_version = m_kvnet.KVNET6
        test_KVNet_test = test_KVNet.test_no_pred_DPV

    elif version_num == 7:
        print("**test with aggregate+R(V7) method")
        model_version = m_kvnet.KVNET7
        test_KVNet_test = test_KVNet.test_no_pred_DPV

    else:
        raise NotImplementedError

    model_KVnet = model_version(feature_dim=args.feature_dim, cam_intrinsics=dataset.cam_intrinsics,
                                d_candi=d_candi, sigma_soft_max=args.sigma_soft_max, KVNet_feature_dim=args.feature_dim,
                                d_upsample_ratio_KV_net=d_upsample, t_win_r=args.t_win, if_refined=True)

    model_KVnet = torch.nn.DataParallel(model_KVnet)
    model_KVnet.cuda()
    total_params = count_parameters(model_KVnet)

    wandb.init(project="stad", name=f'Eval/{exp_name}-{len(traj_Indx)}-gs{global_step}', config=args)
    for model_path in model_paths:
        # model_path_KV = args.model_path
        try:
            global_step = int(os.path.basename(model_path).split('.')[0])
        except:
            global_step = 0

        model_path_KV = model_path
        print('loading KV_net at %s' % (model_path_KV))
        utils_model.load_pretrained_model(model_KVnet, model_path_KV)
        print('Done')
        pred_depths = []
        gt_depths = []
        eval_measures = torch.zeros(10).cuda()

        # with open(args.split_eigen) as f:
        #     lines = f.readlines()
        # eigen_split = [args.dataset_path + '/rawdata/' + line.split(' ')[0] for line in lines]

        for traj_idx in traj_Indx:
            res_fldr = '../results/%s/traj_%d' % (exp_name, traj_idx)
            m_misc.m_makedir(res_fldr)
            scene_path_info = []

            print('Getting the paths for traj_%d' % (traj_idx))
            fldr_path, img_seq_paths, dmap_seq_paths, poses, intrin_path = fun_get_paths(traj_idx)
            dataset.set_paths(img_seq_paths, dmap_seq_paths, poses)

            if args.dataset is 'scanNet':
                # For each trajector in the dataset, we will update the intrinsic matrix #
                dataset.get_cam_intrinsics(intrin_path)

            print('Done')
            dat_array = [dataset[idx] for idx in range(args.t_win * 2 + 1)]
            DMaps_meas = []
            BVs_predict = None
            traj_length = len(dataset)
            print('trajectory length = %d' % (traj_length))

            for frame_cnt, ref_indx in enumerate(range(args.t_win, traj_length - args.t_win - 1)):
                eff_iter = True
                valid_seq = check_datArray_pose(dat_array)

                # Read ref. and src. data in the local time window #
                ref_dat, src_dats = m_misc.split_frame_list(dat_array, args.t_win)

                print(ref_dat['img_path'])
                if frame_cnt == 0:
                    BVs_predict = None

                if valid_seq and eff_iter:
                    # Get poses #
                    src_cam_extMs = m_misc.get_entries_list_dict(src_dats, 'extM')
                    src_cam_poses = \
                        [warp_homo.get_rel_extrinsicM(ref_dat['extM'], src_cam_extM_) \
                         for src_cam_extM_ in src_cam_extMs]

                    src_cam_poses = [
                        torch.from_numpy(pose.astype(np.float32)).cuda().unsqueeze(0)
                        for pose in src_cam_poses]

                    # src_cam_poses size: N V 4 4 #
                    src_cam_poses = torch.cat(src_cam_poses, dim=0).unsqueeze(0)
                    src_frames = [m_misc.get_entries_list_dict(src_dats, 'img')]

                    if frame_cnt == 0 or BVs_predict is None:  # the first window for the traj.
                        BVs_predict_in = None
                    else:
                        BVs_predict_in = BVs_predict

                    print('testing on %d/%d frame in traj %d/%d ... ' % \
                          (frame_cnt + 1, traj_length - 2 * args.t_win, traj_idx + 1, len(traj_Indx)))

                    # set trace for specific frame #

                    BVs_measure, BVs_predict = test_KVNet_test(model_KVnet, d_candi,
                                                               Ref_Dats=[ref_dat],
                                                               Src_Dats=[src_dats],
                                                               Cam_Intrinsics=[dataset.cam_intrinsics],
                                                               t_win_r=args.t_win,
                                                               Src_CamPoses=src_cam_poses,
                                                               BV_predict=BVs_predict_in,
                                                               R_net=True,
                                                               Cam_Intrinsics_imgsize=dataset_imgsize.cam_intrinsics,
                                                               ref_indx=ref_indx)

                    # export_res.export_res_refineNet(ref_dat,  BVs_measure, d_candi_dmap_ref,
                    #                                 res_fldr, ref_indx,
                    #                                 save_mat = True, output_pngs = False, output_dmap_ref=False)
                    dmap = export_res.export_res_img(ref_dat, BVs_measure, d_candi_dmap_ref, res_fldr, frame_cnt)
                    scene_path_info.append([frame_cnt, dataset[ref_indx]['img_path']])

                    pred_depth = dmap
                    try:
                        gt_depth = ref_dat['dmap_imgsize'].squeeze().cpu().detach().numpy()
                    except:
                        print("invalid gt, continue")
                        dat_array.pop(0)
                        dat_array.append(dataset[ref_indx + args.t_win + 1])
                        continue

                    # pred_depths.append(dmap)
                    # gt_depths.append(gt_depth)

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
                    save_folder = save_folder_init + f'-{global_step}'
                    save_folder_png = save_folder_init + f'_png-{global_step}'


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



                elif valid_seq is False:  # if the sequence contains invalid pose estimation
                    BVs_predict = None
                    print('frame_cnt :%d, include invalid poses' % (frame_cnt))

                elif eff_iter is False:
                    BVs_predict = None

                # Update dat_array #
                dat_array.pop(0)
                dat_array.append(dataset[ref_indx + args.t_win + 1])

            m_misc.save_ScenePathInfo('%s/scene_path_info.txt' % (res_fldr), scene_path_info)

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
                   'tot_params': total_params}, step=global_step)


if __name__ == '__main__':
    main()
