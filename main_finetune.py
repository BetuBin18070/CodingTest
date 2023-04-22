import argparse
import datetime
import json
import numpy as np
import os
import time
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import timm

assert timm.__version__ == "0.3.2"  # version check
from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy


from util.datasets import build_dataset
from util.pos_embed import interpolate_pos_embed

from models import models_deit

from engine.engine_finetune import train_one_epoch, evaluate
from util.save_model import save_model

def get_args_parser():
    parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=100, type=int)

    # Model parameters

    parser.add_argument('--input_size', default=32, type=int,
                        help='images input size')


    parser.add_argument('--mlp_ratio', default=4, type=int,
                        help='mlp ratio of FFN')

    parser.add_argument('--norm_pix_loss', default=False, type=int,
                        help='Use (per-patch) normalized pixels as targets for computing loss')

    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # encoder
    parser.add_argument('--patch_size', default=4, type=int,
                        help='Patch size of ViT')
    parser.add_argument('--in_chans', default=3, type=int,
                        help='Number of channels of the image')
    parser.add_argument('--embed_dim', default=192, type=int,
                        help='Patch embedding dim(encoder)')
    parser.add_argument('--depth', default=12, type=int,
                        help='depth of the encoder')
    parser.add_argument('--num_heads', default=3, type=int,
                        help='Number of heads of the encoder')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=5e-5, metavar='LR',
                        help='learning rate (absolute lr 1e-4)')



    # * Finetuning params 
    parser.add_argument('--finetune', default='./output_dir/pretrain/Boot/6k/6/checkpoint-197.pth',
                        help='finetune from checkpoint')
    parser.add_argument('--global_pool', type=bool, default=True,)

    # Dataset parameters

    parser.add_argument('--nb_classes', default=10, type=int,
                        help='number of the classification types')

    parser.add_argument('--output_dir', default='./output_dir/finetune/Boot/6k/6/',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./runs/finetune/Boot/6k/6/',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)


    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--pin_mem', default=True, type=bool,
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')

    return parser


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main(args):
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    setup_seed(seed)

    cudnn.benchmark = True

    dataset_train = build_dataset(is_train=True, args=args)
    dataset_val = build_dataset(is_train=False, args=args)

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    log_writer = SummaryWriter(args.log_dir)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )


    model = models_deit.DistilledVisionTransformer(num_classes=args.nb_classes, img_size=args.input_size,
                                                   patch_size=args.patch_size, embed_dim=args.embed_dim,
                                                   depth=args.depth
                                                   , num_heads=args.num_heads, mlp_ratio=args.mlp_ratio, qkv_bias=True,
                                                   norm_layer=nn.LayerNorm
                                                   ,global_pool=args.global_pool
                                                   ,drop_path_rate=args.drop_path)

    #
    if args.finetune and not args.eval:
        checkpoint = torch.load(args.finetune, map_location='cpu')

        print("Load pre-trained checkpoint from: %s" % args.finetune)
        checkpoint_model = checkpoint['model']

        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

        if args.global_pool:
            assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias',
                                             'fc_norm.weight', 'fc_norm.bias'}
        else:
            assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias'}

        # manually initialize fc layer
        trunc_normal_(model.head.weight, std=0.01)

    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    if args.eval:
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_val_accuracy = 0.0
    max_train_accuracy = 0.0
    for epoch in range(args.epochs):

        train_epoch_loss, train_epoch_acc = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device
        )
        print(f"Accuracy of the network on the {len(dataset_train)} test images: {train_epoch_acc:.1f}%")
        max_train_accuracy = max(max_train_accuracy, train_epoch_acc)
        print(f'Max train accuracy: {max_train_accuracy:.2f}%')

        val_epoch_loss, val_epoch_acc = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {val_epoch_acc:.1f}%")

        if max_val_accuracy<val_epoch_acc:
            save_model(
                args, 666, model, optimizer
            )

        max_val_accuracy = max(max_val_accuracy, val_epoch_acc)
        print(f'Max val accuracy: {max_val_accuracy:.2f}%')

        if log_writer is not None:
            log_writer.add_scalar('finetune/baseline/loss/val', val_epoch_loss, epoch)
            log_writer.add_scalar('finetune/baseline/loss/train', train_epoch_loss, epoch)
            log_writer.add_scalar('finetune/baseline/acc/val', val_epoch_acc, epoch)
            log_writer.add_scalar('finetune/baseline/acc/train', train_epoch_acc, epoch)
            log_writer.flush()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
