#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python main_finetune.py \
  --batch_size=128 \
  --epochs=100 \
  --input_size=32 \
  --drop_path=0.1 \
  --mlp_ratio=4 \
  --patch_size=4 \
  --in_chans=3 \
  --embed_dim=192 \
  --depth=12 \
  --num_heads=3 \
  --weight_decay=0.05 \
  --lr=5e-5 \
  --finetune='./output_dir/pretrain/Boot/6k/6/checkpoint-197.pth' \
  --nb_classes=10 \
  --output_dir='./output_dir/finetune/Boot/6k/' \
  --log_dir='./runs/finetune/Boot/6k/' \
  --device='cuda' \
  --seed=0 \
  --num_workers=4 \
  