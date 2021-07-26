#!/bin/bash

EXP_FOLDER='exp_jul14_seed42'
RANDOM_SEED=42

## fine-tuned CLIP classifiers
CUDA_VISIBLE_DEVICES=0 python train.py train.random_seed=$RANDOM_SEED train.exps_folder=$EXP_FOLDER train.feats_backbone=clip train.max_epochs=50 train.aggregator.type=maxpool train.load_from_last_ckpt=False &
CUDA_VISIBLE_DEVICES=0 python train.py train.random_seed=$RANDOM_SEED train.exps_folder=$EXP_FOLDER train.feats_backbone=clip train.max_epochs=50 train.aggregator.type=random_index train.load_from_last_ckpt=False &
CUDA_VISIBLE_DEVICES=0 python train.py train.random_seed=$RANDOM_SEED train.exps_folder=$EXP_FOLDER train.feats_backbone=clip train.max_epochs=50 train.aggregator.type=two_random_index train.load_from_last_ckpt=False &

## zero-shot CLIP classifiers
CUDA_VISIBLE_DEVICES=0 python train.py train.random_seed=$RANDOM_SEED train.model=zero_shot_cls train.exps_folder=$EXP_FOLDER train.feats_backbone=clip train.max_epochs=1 train.aggregator.type=maxpool train.load_from_last_ckpt=False train.max_epochs=50 &
CUDA_VISIBLE_DEVICES=0 python train.py train.random_seed=$RANDOM_SEED train.model=zero_shot_cls train.exps_folder=$EXP_FOLDER train.feats_backbone=clip train.max_epochs=1 train.aggregator.type=random_index train.load_from_last_ckpt=False train.max_epochs=50 &
CUDA_VISIBLE_DEVICES=0 python train.py train.random_seed=$RANDOM_SEED train.model=zero_shot_cls train.exps_folder=$EXP_FOLDER train.feats_backbone=clip train.max_epochs=1 train.aggregator.type=two_random_index train.load_from_last_ckpt=False train.max_epochs=50

