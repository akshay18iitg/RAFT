'''
Copyright (c) 2020-2021, Christian Lagemann
'''

import os
import argparse
import torch.multiprocessing as mp
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import time
from tqdm import tqdm
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.signal
import scipy.io
from numpy import genfromtxt

from flowNetsRAFT import RAFT
from getIP import resolve_master_node
import socket
import ifcfg

from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from nvidia.dali.plugin.pytorch import DALIClassificationIterator as PyTorchIterator
import nvidia.dali.tfrecord as tfrec

from subprocess import call
import os.path

###############################################################################
class TFRecordPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, num_gpus, shard_id, num_shards, tfrecord, tfrecord_idx, exec_pipelined=False, exec_async=False, is_shuffle=False, image_shape=[2,32,32], label_shape=[2,12]):
        super(TFRecordPipeline, self).__init__(batch_size, num_threads, device_id, exec_pipelined=False, exec_async=False)
        self.input = ops.TFRecordReader(path = tfrecord, 
                                        index_path = tfrecord_idx,
                                        random_shuffle=is_shuffle,
                                        pad_last_batch = True,
                                        shard_id=shard_id,
                                        num_shards=num_shards,
                                        features = {"target" : tfrec.FixedLenFeature([], tfrec.string, ""),
                                                    "label": tfrec.FixedLenFeature([], tfrec.string,  ""),
                                                    "flow" : tfrec.FixedLenFeature([], tfrec.string, ""),
                                                   })
    
        self.decode = ops.PythonFunction(function=self.extract_view, num_outputs=1)
        self.reshape_image = ops.Reshape(shape=image_shape)
        self.reshape_label = ops.Reshape(shape=label_shape)

    def extract_view(self, data):
        ext_data = data.view('<f4')
        return ext_data

    def define_graph(self):
        inputs = self.input(name="Reader")
        images = self.reshape_image(self.decode(inputs['target']))
        labels = self.reshape_label(self.decode(inputs['label']))
        flows = self.reshape_image(self.decode(inputs['flow']))

        return images, labels, flows

### spline windowing
def _spline_window(window_size, power=2):
    """
    Squared spline (power=2) window function:
    """
    intersection = int(window_size/4)
    wind_outer = (abs(2*(scipy.signal.triang(window_size))) ** power)/2
    wind_outer[intersection:-intersection] = 0

    wind_inner = 1 - (abs(2*(scipy.signal.triang(window_size) - 1)) ** power)/2
    wind_inner[:intersection] = 0
    wind_inner[-intersection:] = 0

    wind = wind_inner + wind_outer
    wind = wind / np.average(wind)
    return wind

#spline windowing
cached_2d_windows = dict()
def _window_2D(window_size, power=2):
    """
    Make a 1D window function, then infer and return a 2D window function.
    Done with an augmentation, and self multiplication with its transpose.
    Could be generalized to more dimensions.
    """
    # Memoization
    global cached_2d_windows
    key = "{}_{}".format(window_size, power)
    if key in cached_2d_windows:
        wind = cached_2d_windows[key]
    else:
        wind = _spline_window(window_size, power)
        wind = np.expand_dims(np.expand_dims(wind, -1), -1)
        wind = wind * wind.transpose(1, 0, 2)
        cached_2d_windows[key] = wind
    return wind

###############################################################################

try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass

### main method
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1, type=int,
                        help='number of compute nodes')
    parser.add_argument('-g', '--gpus', default=2, type=int,
                        help='number of gpus per node')
    parser.add_argument('--name', type=str, default='RAFT-PIV32_test_backstep',
                        help='name of experiment')
    parser.add_argument('--input_path_ckpt', type=str,
                        default='./precomputed_ckpts/RAFT32-PIV_ProbClass1/ckpt.tar',
                        help='path of already trained checkpoints')
    parser.add_argument('--recover', type=eval, default=True,
                        help='Wether to load an existing checkpoint')
    parser.add_argument('--output_dir_results', type=str, default='./test_results/',
                        help='output directory of test results')

    parser.add_argument('--test_dataset', type=str, default='backstep',
                        choices=['backstep', 'cylinder', 'jhtdb', 'dns_turb', 'sqg', 'tbl', 'twcf'],
                        help='test dataset to evaluate')
    parser.add_argument('--plot_results', type=eval, default=True,
                        help="""Whether or not to plot predicted results.""")

    parser.add_argument('--amp', type=eval, default=False, help='Whether to use auto mixed precision')
    parser.add_argument('-a', '--arch', type=str, default='RAFT32', choices=['RAFT32'],
                        help='Type of architecture to use')
    parser.add_argument('--batch_size_test', default=1, type=int)
    parser.add_argument('--split_size', default=50, type=int)
    parser.add_argument('--offset', default=32, type=int,
                        help='interrogation window size')
    parser.add_argument('--shift', default=8, type=int,
                        help='shift of interrogation window in px')

    parser.add_argument('--iters', default=12, type=int,
                        help='number of update steps in ConvGRU')
    args = parser.parse_args()
    print('args parsed')

    if args.test_dataset == 'tbl':
        args.image_height = 256
        args.image_width = 3296
    elif args.test_dataset == 'twcf':
        args.image_height = 2160
        args.image_width = 2560
    else:
        args.image_height = 256
        args.image_width = 256

    mp.spawn(train, nprocs=args.gpus, args=(args,))

def train(GPU,args):
    #init procedure
    torch.manual_seed(0)
    print('getting master_addr', flush=True)
    if "SLURM_JOBID" in os.environ:
        masterIP, _, _, _, _, _, _ = resolve_master_node(platform.node(), 8888)
        os.environ['MASTER_ADDR'] = masterIP
    else:
        os.environ['MASTER_ADDR'] = '127.0.0.1'

    print('start GPU:' + str(GPU))

    MASTER_PORT = int(os.environ.get("MASTER_PORT", 8738))
    MASTER_ADDR = os.environ.get("MASTER_ADDR")
    N_NODES = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NNODES", 1)))
    NODE_RANK = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))
    WORLD_SIZE = args.gpus * N_NODES
    rank = NODE_RANK * args.gpus + GPU
    backend = 'nccl'
    NODE_NAME = socket.gethostname()
    NODE_IP = socket.gethostbyname(NODE_NAME)

    print('node_name', NODE_NAME, 'node_ip', NODE_IP, 'rank', rank, 'node_rank', NODE_RANK, 'GPU', GPU, 'master_IP', MASTER_ADDR, 'master_port', MASTER_PORT, 'world_size    ', WORLD_SIZE, flush=True)

    tcp_store = dist.TCPStore(MASTER_ADDR, MASTER_PORT, WORLD_SIZE, rank == 0)
    dist.init_process_group(backend,
                            store=tcp_store,
                            rank=rank,
                            world_size=WORLD_SIZE
                            )

    int_GPU = GPU
    GPU = torch.device("cuda", GPU)
    torch.cuda.set_device(GPU)

    print('synchronizing all processes', flush=True)
    dist.barrier()
    print('processes synchronized', flush=True)

    # create a second gloo group
    print('creating second process group', flush=True)
    list_ranks = [int(_) for _ in range(dist.get_world_size())]
    gather_group = dist.new_group(ranks=list_ranks, backend='gloo')

    if args.arch == 'RAFT32':
        model = RAFT()
        print('Selected model: Standard RAFT - -', args.arch)
    else:
        raise ValueError('Selected model not supported: ', args.arch)

    pytorch_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of trainable parameters: ', pytorch_trainable_params)

    output_dir = args.output_dir_results + args.name
    if rank == 0:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    if args.recover:
        print('recovering: ', args.input_path_ckpt)
        checkpoint = torch.load(args.input_path_ckpt)
        model.load_state_dict(checkpoint['model_state_dict'],strict=False)
        print('model recovered')
    
    model.cuda(GPU)
    model = DDP(model, device_ids=[GPU],find_unused_parameters=True)

    # dataset
    if args.test_dataset == 'backstep':
        print('backstep dataset loaded', flush=True)
        # backstep test case
        test_tfrecord = '../data/Test_Dataset_10Imgs_backstep.tfrecord-00000-of-00001'
        test_tfrecord_idx = "../data/idx_files/Test_Dataset_10Imgs_backstep.idx"
    elif args.test_dataset == 'cylinder':
        print('cylinder dataset loaded', flush=True)
        # cylinder test case
        test_tfrecord = '../data/Test_Dataset_10Imgs_cylinder.tfrecord-00000-of-00001'
        test_tfrecord_idx = "../data/idx_files/Test_Dataset_10Imgs_cylinder.idx"
    elif args.test_dataset == 'jhtdb':
        print('jhtdb dataset loaded', flush=True)
        # jhtdb test case
        test_tfrecord = '../data/Test_Dataset_10Imgs_jhtdb.tfrecord-00000-of-00001'
        test_tfrecord_idx = "../data/idx_files/Test_Dataset_10Imgs_jhtdb.idx"
    elif args.test_dataset == 'dns_turb':
        print('dns-turbulence dataset loaded', flush=True)
        # dns turbulence test case
        test_tfrecord = '../data/Test_Dataset_10Imgs_dns_turb.tfrecord-00000-of-00001'
        test_tfrecord_idx = "../data/idx_files/Test_Dataset_10Imgs_dns_turb.idx"
    elif args.test_dataset == 'sqg':
        print('sqg dataset loaded', flush=True)
        # sqg test case
        test_tfrecord = '../data/Test_Dataset_10Imgs_sqg.tfrecord-00000-of-00001'
        test_tfrecord_idx = "../data/idx_files/Test_Dataset_10Imgs_sqg.idx"
    elif args.test_dataset == 'tbl':
        print('TBL dataset loaded', flush=True)
        # DNS transitional TBL
        test_tfrecord = '../data/Dataset_TransTBL_Original8px_fullFrame_withGT.tfrecord-00000-of-00001'
        test_tfrecord_idx = "../data/idx_files/Dataset_TransTBL_Original8px_withGT_fullFrame.idx"
    elif args.test_dataset == 'twcf':
        print('TWCF dataset loaded', flush=True)
        # experimental data turbulent wavy channel flow
        test_tfrecord = '../data/Test_Dataset_AR_rawImage.tfrecord-00000-of-00001'
        test_tfrecord_idx = "../data/idx_files/Test_Dataset_AR_rawImage.idx"
    else:
        raise ValueError('Selected test dataset not available: ', args.test_dataset)

    # DALI data loading
    tfrecord2idx_script = "tfrecord2idx"
    if not os.path.isfile(test_tfrecord_idx):
        call([tfrecord2idx_script, test_tfrecord, test_tfrecord_idx])

    test_pipe = TFRecordPipeline(batch_size=args.batch_size_test, num_threads=8, device_id=int_GPU, num_gpus=1,
                                  tfrecord=test_tfrecord, tfrecord_idx=test_tfrecord_idx,
                                  num_shards=WORLD_SIZE, shard_id=rank,
                                  is_shuffle=False, image_shape=[2, args.image_height, args.image_width], label_shape=[12, ])
    test_pipe.build()
    test_pii = DALIGenericIterator(test_pipe, ['target', 'label', 'flow'],
                                   size=int(test_pipe.epoch_size("Reader") / WORLD_SIZE),
                                   last_batch_padded=True, fill_last_batch=False, auto_reset=True)

    # Testing using full frame folding
    with torch.set_grad_enabled(False):
        if rank == 0:
            print('################################# Start Testing #####################################')  #

        model.eval()
        test_loader_len = int(math.ceil(test_pii._size / args.batch_size_test))
        test_pbar = tqdm(enumerate(test_pii), total=test_loader_len, leave=False)

        total_test_samples = 0
        sum_test_epe = 0.0

        # store resuls
        results = np.zeros((test_pii._size, 4, args.image_height, args.image_width))
        epe_array = np.zeros((test_pii._size))

        # compute b-spline window
        WINDOW_SPLINE_2D = torch.from_numpy(np.squeeze(_window_2D(window_size=args.offset, power=2)))

        # start time
        start_time = time.time()

        #load PIV data
        if args.test_dataset == 'twcf':
            PIV_results_TWCF = np.load('./data/PIV_results_TWCF.npy')
            mask_TWCF = np.load('./data/mask_TWCF.npy')

        for i_batch, sample_batched in test_pbar:
            t0 = time.time()
            local_dict = sample_batched[0]
            images = local_dict['target'].type(torch.FloatTensor).cuda(GPU) / 256
            flows = local_dict['flow'].type(torch.FloatTensor).cuda(GPU)

            folding_mask = torch.ones_like(images)

            #compute number of patches for folding operation
            B, C, H, W = images.size()
            NUM_Yvectors, NUM_Xvectors, numImages = int(H / args.shift - (args.offset / args.shift - 1)), \
                                                    int(W / args.shift - (args.offset / args.shift - 1)), \
                                                    test_pii._size

            # allocate memory of predicted images
            predicted_flows = torch.zeros_like(images).cuda()

            # create patches of image and flow
            patches = images.unfold(3, args.offset, args.shift).unfold(2, args.offset, args.shift).permute(0, 2, 3, 1, 5, 4)
            patches = patches.reshape((-1, 2, args.offset, args.offset))
            flow_patches = flows.unfold(3, args.offset, args.shift).unfold(2, args.offset, args.shift).permute(0, 2, 3, 1, 5, 4)
            flow_patches = flow_patches.reshape((-1, 2, args.offset, args.offset))
            splitted_patches = torch.split(patches, args.split_size, dim=0)
            splitted_flow_patches = torch.split(flow_patches, args.split_size, dim=0)

            # Forward pass
            with autocast(enabled=args.amp):
                #unfold flow
                predicted_flow_patches = predicted_flows.unfold(3, args.offset, args.shift)\
                    .unfold(2, args.offset, args.shift).permute(0, 2, 3, 1, 5, 4)
                predicted_flow_patches = predicted_flow_patches.reshape((-1, 2, args.offset, args.offset))
                #split flow patch tensor in batches for evaluation
                splitted_predicted_flow_patches = torch.split(predicted_flow_patches, args.split_size, dim=0)
                splitted_flow_output_patches = []

                for split in range(len(splitted_patches)):
                    pred_flows = model(splitted_patches[split], splitted_flow_patches[split], \
                                       flow_init=splitted_predicted_flow_patches[split], args=args)
                    all_flow_iters = pred_flows[0]
                    splitted_flow_output_patches.append(all_flow_iters[-1])

                #fold and weight predicted flow patches
                flow_output_patches = torch.cat(splitted_flow_output_patches, dim=0)
                flow_output_patches = flow_output_patches * WINDOW_SPLINE_2D.cuda()
                flow_output_patches = flow_output_patches.reshape(
                    (B, NUM_Yvectors, NUM_Xvectors, 2, args.offset, args.offset)).permute(0, 3, 1, 2, 4, 5)
                flow_output_patches = flow_output_patches.contiguous().view(B, C, -1, args.offset * args.offset)
                flow_output_patches = flow_output_patches.permute(0, 1, 3, 2)
                flow_output_patches = flow_output_patches.contiguous().view(B, C * args.offset * args.offset, -1)
                predicted_flows_iter = F.fold(flow_output_patches, output_size=(H, W), kernel_size=args.offset,
                                              stride=args.shift)

                #compute folding mask
                mask_patches = folding_mask.unfold(3, args.offset, args.shift).unfold(2, args.offset, args.shift)
                mask_patches = mask_patches.contiguous().view(B, C, -1, args.offset, args.offset)
                mask_patches = mask_patches * WINDOW_SPLINE_2D.cuda()
                mask_patches = mask_patches.view(B, C, -1, args.offset * args.offset)
                mask_patches = mask_patches.permute(0, 1, 3, 2)
                mask_patches = mask_patches.contiguous().view(B, C * args.offset * args.offset, -1)
                folding_mask = F.fold(mask_patches, output_size=(H, W), kernel_size=args.offset, stride=args.shift)

                predicted_flows += predicted_flows_iter / folding_mask

                # compute epe
                test_epe_loss = torch.sum((predicted_flows[:, :, :,:] - flows[:, :, :,:]) ** 2,
                                          dim=1).sqrt().view(-1).mean().item()
                epe_array[i_batch*B:i_batch*B+B] = test_epe_loss
                total_test_samples += B
                sum_test_epe += test_epe_loss * B
                total_test_epe_loss = sum_test_epe / total_test_samples

                #store results
                flow_name = output_dir + '/' + 'Rank_{:02d}'.format(rank) + 'Test_image_{:03d}'.format(i_batch) + '.png'
                u_plot_pred = torch.squeeze(predicted_flows[:, 0, :, :]).cpu().numpy()
                v_plot_pred = torch.squeeze(predicted_flows[:, 1, :, :]).cpu().numpy()
                u_plot_gt = torch.squeeze(flows[:, 0, :, :]).cpu().numpy()
                v_plot_gt = torch.squeeze(flows[:, 1, :, :]).cpu().numpy()

                results[i_batch*B:i_batch*B+B, 0, :, :] = u_plot_pred
                results[i_batch*B:i_batch*B+B, 1, :, :] = v_plot_pred
                results[i_batch*B:i_batch*B+B, 2, :, :] = u_plot_gt
                results[i_batch*B:i_batch*B+B, 3, :, :] = v_plot_gt

                if args.plot_results:
                    if args.test_dataset == 'twcf':
                        U_PascalPIV = PIV_results_TWCF[numImages * rank + i_batch, 0, :, :]
                        V_PascalPIV = PIV_results_TWCF[numImages * rank + i_batch, 1, :, :]
                        # plot figure
                        plt.figure(num=None, figsize=(24, 16), dpi=120, facecolor='w', edgecolor='k')
                        plt.subplot(2, 2, 1)
                        plt.pcolor(np.squeeze(U_PascalPIV), cmap='Greys', vmin=-2, vmax=12)
                        plt.axis('off')
                        cbar = plt.colorbar()
                        cbar.ax.set_ylabel('displacement [px]', fontsize=14)
                        t = plt.text(0, 505, 'PascalPIV', fontsize=16)
                        t.set_bbox(dict(facecolor='white', alpha=1, edgecolor='white'))
                        plt.subplot(2, 2, 2)
                        plt.pcolor(np.squeeze(V_PascalPIV), cmap='Greys', vmin=-1, vmax=1)
                        plt.axis('off')
                        cbar = plt.colorbar()
                        cbar.ax.set_ylabel('displacement [px]', fontsize=14)
                        plt.subplot(2, 2, 3)
                        plt.pcolor(u_plot_pred * mask_TWCF, cmap='Greys', vmin=-2, vmax=12)
                        plt.axis('off')
                        cbar = plt.colorbar()
                        cbar.ax.set_ylabel('displacement [px]', fontsize=14)
                        t = plt.text(0, 2025, 'RAFT32-PIV', fontsize=16)
                        t.set_bbox(dict(facecolor='white', alpha=1, edgecolor='white'))
                        plt.subplot(2, 2, 4)
                        plt.pcolor(v_plot_pred * mask_TWCF, cmap='Greys', vmin=-1, vmax=1)
                        plt.axis('off')
                        cbar = plt.colorbar()
                        cbar.ax.set_ylabel('displacement [px]', fontsize=14)
                        plt.savefig(flow_name)
                        plt.close()
                    elif args.test_dataset == 'tbl':
                        # plot figure
                        plt.figure(num=None, figsize=(24, 16), dpi=120, facecolor='w', edgecolor='k')
                        plt.subplot(2, 2, 1)
                        plt.pcolor(np.squeeze(u_plot_pred), cmap='Greys', vmin=2, vmax=8)
                        plt.axis('off')
                        cbar = plt.colorbar()
                        cbar.ax.set_ylabel('displacement [px]', fontsize=14)
                        t = plt.text(0, 225, 'RAFT32-PIV', fontsize=16)
                        t.set_bbox(dict(facecolor='white', alpha=1, edgecolor='white'))
                        plt.subplot(2, 2, 2)
                        plt.pcolor(np.squeeze(v_plot_pred), cmap='Greys', vmin=-0.5, vmax=0.5)
                        plt.axis('off')
                        cbar = plt.colorbar()
                        cbar.ax.set_ylabel('displacement [px]', fontsize=14)
                        plt.subplot(2, 2, 3)
                        plt.pcolor(u_plot_gt, cmap='Greys', vmin=2, vmax=8)
                        plt.axis('off')
                        cbar = plt.colorbar()
                        cbar.ax.set_ylabel('displacement [px]', fontsize=14)
                        t = plt.text(0, 225, 'ground truth', fontsize=16)
                        t.set_bbox(dict(facecolor='white', alpha=1, edgecolor='white'))
                        plt.subplot(2, 2, 4)
                        plt.pcolor(v_plot_gt, cmap='Greys', vmin=-0.5, vmax=0.5)
                        plt.axis('off')
                        cbar = plt.colorbar()
                        cbar.ax.set_ylabel('displacement [px]', fontsize=14)
                        plt.savefig(flow_name)
                        plt.close()
                    else:
                        minValU, maxValU = -4, 4
                        minValV, maxValV = -4, 4

                        # plot figure
                        plt.figure(num=None, figsize=(24, 16), dpi=120, facecolor='w', edgecolor='k')
                        plt.subplot(3, 2, 1)
                        plt.pcolor(u_plot_pred, cmap='Greys', vmin=minValU, vmax=maxValU)
                        plt.axis('off')
                        cbar = plt.colorbar()
                        cbar.ax.set_ylabel('displacement [px]',fontsize=14)
                        t = plt.text(0,238,'RAFT32-PIV',fontsize=16)
                        t.set_bbox(dict(facecolor='white', alpha=1.0, edgecolor='white'))
                        plt.subplot(3, 2, 3)
                        plt.pcolor(u_plot_gt, cmap='Greys', vmin=minValU, vmax=maxValU)
                        plt.axis('off')
                        cbar = plt.colorbar()
                        cbar.ax.set_ylabel('displacement [px]', fontsize=14)
                        t = plt.text(0, 238, 'ground truth', fontsize=16)
                        t.set_bbox(dict(facecolor='white', alpha=1.0, edgecolor='white'))
                        plt.subplot(3, 2, 5)
                        plt.pcolor(u_plot_pred - u_plot_gt, cmap='bwr', vmin=-0.25, vmax=0.25)
                        plt.axis('off')
                        cbar = plt.colorbar()
                        cbar.ax.set_ylabel('abs. error [px]', fontsize=14)
                        t = plt.text(0, 238, 'absolute error', fontsize=16)
                        t.set_bbox(dict(facecolor='white', alpha=1.0, edgecolor='white'))
                        plt.subplot(3, 2, 2)
                        plt.pcolor(v_plot_pred, cmap='Greys', vmin=minValV, vmax=maxValV)
                        plt.axis('off')
                        cbar = plt.colorbar()
                        cbar.ax.set_ylabel('displacement [px]', fontsize=14)
                        plt.subplot(3, 2, 4)
                        plt.pcolor(v_plot_gt, cmap='Greys', vmin=minValV, vmax=maxValV)
                        plt.axis('off')
                        cbar = plt.colorbar()
                        cbar.ax.set_ylabel('displacement [px]', fontsize=14)
                        plt.subplot(3, 2, 6)
                        plt.pcolor(v_plot_pred - v_plot_gt, cmap='bwr', vmin=-0.25, vmax=0.25)
                        plt.axis('off')
                        cbar = plt.colorbar()
                        cbar.ax.set_ylabel('abs. error [px]', fontsize=14)
                        plt.savefig(flow_name)
                        plt.close()

                print('Total test samples: ', total_test_samples, ' test epe loss: ', total_test_epe_loss, \
                      ' rank: ', rank, ' required time: ', time.time() - t0,
                      flush=True)

        #gather results of all gpus
        results = torch.tensor(results)
        epe_array = torch.tensor(epe_array)
        result_list = [torch.ones_like(results) for _ in range(dist.get_world_size())]
        epe_list = [torch.ones_like(epe_array) for _ in range(dist.get_world_size())]

        if rank == 0:
            torch.distributed.gather(tensor=results, gather_list=result_list, dst=0, group=gather_group)
        else:
            torch.distributed.gather(tensor=results, dst=0, group=gather_group)

        if rank == 0:
            torch.distributed.gather(tensor=epe_array, gather_list=epe_list, dst=0, group=gather_group)
        else:
            torch.distributed.gather(tensor=epe_array, dst=0, group=gather_group)


        if rank == 0:
            predictions = torch.cat(result_list,dim=0)
            mean_epe = torch.mean(torch.cat(epe_list, dim=0))
            print('test case ', args.test_dataset, ' mean epe value: ', mean_epe)

            save_name = output_dir + '/' + 'results' + '.npy'
            np.save(save_name, predictions)
            print('overall execution time: ', time.time() - start_time, flush=True)

if __name__ == '__main__':
    main()
