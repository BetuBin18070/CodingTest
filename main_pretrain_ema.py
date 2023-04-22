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
import torchvision.transforms as transforms
import torchvision
import copy

import timm
from timm.models.layers import trunc_normal_
assert timm.__version__ == "0.3.2"  # version check

from functools import partial

from models import models_mae_Boost

from models import models_mae_Byol


import util.criterions

from engine.engine_pretrain_Byol import train_one_epoch

from util.save_model import save_model


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=2048, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=200, type=int)

    # Model parameters
    parser.add_argument('--input_size', default=32, type=int,
                        help='images input size')
    parser.add_argument('--mlp_ratio', default=4, type=int,
                        help='mlp ratio of FFN')
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

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
    # decoder
    parser.add_argument('--decoder_embed_dim', default=192, type=int,
                        help='Patch embedding dim(decoder)')
    parser.add_argument('--decoder_depth', default=5, type=int,
                        help='depth of the decoder')
    parser.add_argument('--decoder_num_heads', default=3, type=int,
                        help='Number of heads of the decoder')

    # Knowledge distillation
    # teacher
    parser.add_argument('--first_MAE', default='output_dir/pretrain/baseline/epoch34/checkpoint-33.pth',
                        help='finetune from checkpoint')

    parser.add_argument('--start_epoch', default=40, type=int,
                        help='start epoch')

    parser.add_argument('--output_feature_layers_start', default=12, type=int,
                        help='Output feature layers of teacher')
    parser.add_argument('--output_feature_layers_end', default=12, type=int,
                        help='Output feature layers of teacher')

    parser.add_argument('--moving_average_decay', default=0.999999, type=float,
                        help='moving average decay of EMA')

    parser.add_argument('--is_BatchNorm', default=False, type=bool,
                        help='BatchNorm as the Decoder norm')

    parser.add_argument('--lr_start_from_scratch', default=False, type=bool,
                        help='lr Starting from scratch')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (absolute lr)')

    # Dataset parameters
    parser.add_argument('--data_path', default='./data', type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default='./output_dir/pretrain/ema/start34/',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./runs/pretrain/ema/start34/',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.set_defaults(pin_mem=True)

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

    seed = args.seed
    setup_seed(seed)

    cudnn.benchmark = True

    # simple augmentation
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    dataset_train = torchvision.datasets.CIFAR10(root=args.data_path, train=True,
                                                 download=True, transform=transform_train)
    print(dataset_train)

    sampler_train = torch.utils.data.RandomSampler(dataset_train)

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

    # define the model
    model_student = models_mae_Boost.MaskedAutoencoderDeiT(img_size=args.input_size, patch_size=args.patch_size,
                                                           in_chans=args.in_chans,
                                                           embed_dim=args.embed_dim, depth=args.depth,
                                                           num_heads=args.num_heads,
                                                           decoder_embed_dim=args.decoder_embed_dim,
                                                           decoder_depth=args.decoder_depth
                                                           , decoder_num_heads=args.decoder_num_heads,
                                                           mlp_ratio=args.mlp_ratio,
                                                           output_feature_layers_start=args.output_feature_layers_start,
                                                           output_feature_layers_end=args.output_feature_layers_end,
                                                           output_dim=args.embed_dim,
                                                           is_teacher=False, norm_layer=partial(nn.LayerNorm, eps=1e-6)
                                                           , norm_pix_loss=args.norm_pix_loss
                                                           ,is_BatchNorm=args.is_BatchNorm)


    checkpoint = torch.load(args.first_MAE, map_location='cpu')

    checkpoint_model = checkpoint["model"]

    del checkpoint_model["decoder_pred.weight"]
    del checkpoint_model["decoder_pred.bias"]

    model_student.load_state_dict(checkpoint_model, strict=False)

    trunc_normal_(model_student.decoder_pred.weight, std=0.01)



    model_student.to(device)

    

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    # define teacher
    model_teacher = copy.deepcopy(model_student)
    model_teacher.eval().requires_grad_(False)
    model_teacher.to(device)
    model_teacher.is_teacher = True


  
    model_byol=models_mae_Byol.BYOL(
        teacher=model_teacher,
        student=model_student,
        moving_average_decay=args.moving_average_decay,
        norm_pix_loss=args.norm_pix_loss
    )

    model_byol.to(device)

    optimizer = torch.optim.AdamW(model_byol.student.parameters(), lr=args.lr, betas=(0.9, 0.95),
                                  weight_decay=args.weight_decay)

    print(optimizer)
    for epoch in range(args.start_epoch,args.epochs):


        epoch_loss = train_one_epoch(
            model_byol, data_loader_train,
            optimizer, device, epoch,
            log_writer=log_writer,
            args=args
        )

        if log_writer is not None:
            log_writer.add_scalar('/pretrain/baseline/loss', epoch_loss, epoch)
            log_writer.flush()
        if epoch == args.epochs -1:
            save_model(
                args, epoch, model_byol.student, optimizer
            )
            
            save_model(
                args, epoch+1000, model_byol.teacher, optimizer
            )


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
