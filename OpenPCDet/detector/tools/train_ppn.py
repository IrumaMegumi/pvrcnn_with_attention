import argparse
import datetime
import glob
import os
from pathlib import Path
from test import repeat_eval_ckpt

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

from pcdet.config import cfg, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models.backbones_3d.pfe.PointProposal import PointProposalNet_v2
from pcdet.utils import common_utils
from tools.train_utils.train_ppn_utils import ppn_train_model
from tqdm import tqdm

#要测一下训练时间
def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--batch_size', type=int, default=4, required=False, help='batch size for training')
    parser.add_argument('--epochs',type=int,default=30,required=False, help='number of epochs to train for')
    parser.add_argument('--workers',type=int, default=0, help='number of workers for dataloader')
    parser.add_argument('--num_object_points',type=int,default=6000,help='number of points you selected from original points, 6000 is the number for kitti dataset')
    parser.add_argument('--num_keypoints',type=int,default=2048, help='number of keypoints you want to get')
    
    #useless parameters
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--merge_all_iters_to_one_epoch', action='store_true', default=False, help='')
    # TODO: it is not elegant
    parser.add_argument('--cfg_file',type=str, default=None, help='specify the config for training')
    args = parser.parse_args()
    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    return args, cfg

def main():
    #注意观察这里的参数有无变动
    # TODO：填写pointpainting的yaml
    args, cfg = parse_config()
    if getattr(args, 'launcher', None) == 'none':
        dist_train = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_train = True

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus

    args.epochs = cfg.OPTIMIZATION.NUM_EPOCHS if args.epochs is None else args.epochs

    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / 'point_proposal_network' / args.extra_tag
    ckpt_dir = output_dir / 'ckpt'
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / ('log_train_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    if dist_train:
        logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)
    if cfg.LOCAL_RANK == 0:
        os.system('cp %s %s' % (args.cfg_file, output_dir))

    tb_log = SummaryWriter(log_dir=str(output_dir / 'tensorboard')) if cfg.LOCAL_RANK == 0 else None

    # -----------------------create dataloader & network & optimizer---------------------------
    #create train dataloader
    train_set, train_loader, train_sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_train, workers=args.workers,
        logger=logger,
        training=True,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
        total_epochs=args.epochs
    )

    #训练集模型输出探针检查
    # for data_dict in tqdm(train_loader,desc="Loading training data",leave=False):
    #     pass

    # create test dataloader
    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_train, workers=args.workers, logger=logger, training=False
    )
    #测试集模型输出探针检查
    # for data_dict in tqdm(test_loader,desc="Loading training data",leave=False):
    #     pass
    
    model = PointProposalNet_v2(num_object_points=args.num_object_points, num_keypoints=args.num_keypoints)
    model.cuda()

    optimizer = optim.Adam(model.parameters(),lr=0.001, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20*len(train_loader), gamma=0.5)

    model.train()  # before wrap to DistributedDataParallel to support fixed some parameters
    if dist_train:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[cfg.LOCAL_RANK % torch.cuda.device_count()])
    logger.info(model)

    # -----------------------start training---------------------------
    logger.info('**********************Start training %s/%s(%s)**********************'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))

    ## TODO: 写训练部分
    loss=float('inf')
    start_epoch=0
    start_iter=0
    total_epochs=args.epochs
    rank=cfg.LOCAL_RANK
    ppn_train_model(
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        train_loader=train_loader,
        test_loader=test_loader,
        start_iter=start_iter,
        start_epoch=start_epoch,
        total_epochs=total_epochs,
        train_sampler=train_sampler,
        rank=cfg.LOCAL_RANK,
        tb_log=tb_log,
        ckpt_save_dir=ckpt_dir
    )

    logger.info('**********************End training %s/%s(%s)**********************\n\n\n'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))

if __name__ == '__main__':
    main()

