#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python main_pretrain.py \
  --batch_size=512 \
  --epochs=34 \
  --input_size=32 \
  --mlp_ratio=4 \
  --mask_ratio=0.75 \
  --patch_size=4 \
  --in_chans=3 \
  --embed_dim=192 \
  --depth=12 \
  --num_heads=3 \
  --decoder_embed_dim=192 \
  --decoder_depth=5 \
  --decoder_num_heads=3 \
  --weight_decay=0 \
  --lr=1e-3 \
  --data_path='./data' \
  --output_dir='./output_dir/pretrain/baseline/epoch34/' \
  --log_dir='./runs/pretrain/baseline/epoch34/' \
  --device='cuda' \
  --seed=0 \
  --num_workers=4