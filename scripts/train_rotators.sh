#!/bin/bash

EXP_FOLDER='exp_jul14_seed42'
PRETRAINED_CLS="<root_path>/clip-single_cls-random_index/checkpoints/<ckpt_name>.ckpt'"
RANDOM_SEED=42

## LAGOR rotator
CUDA_VISIBLE_DEVICES=0 python train.py train.random_seed=$RANDOM_SEED train.exps_folder=$EXP_FOLDER train.feats_backbone=clip train.model=rotator_sl_direct train.max_epochs=50 train.aggregator.type=random_index train.batch_size=64 train.load_from_last_ckpt=False train.rotator.pretrained_cls=$PRETRAINED_CLS train.lr=5e-5 train.loss.cls_weight=0.2