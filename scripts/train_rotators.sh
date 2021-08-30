#!/bin/bash

## LAGOR rotators for seeds 0 thru 9.

## Seed 0
RANDOM_SEED=0
PRETRAINED_CLS="'/home/jdtho/snare/clips/exp_aug23_seed_0/clip-single_cls-two_random_index/checkpoints/epoch=0024-val_acc=0.81710.ckpt'"
EXP_FOLDER="rotators/exp_aug28_est_init_final_seed_$RANDOM_SEED"
CUDA_VISIBLE_DEVICES=0 python train.py train.random_seed=$RANDOM_SEED train.exps_folder=$EXP_FOLDER train.feats_backbone=clip train.model=rotator train.max_epochs=50 train.aggregator.type=two_random_index train.rotator.pretrained_cls=$PRETRAINED_CLS train.lr=5e-5 &
EXP_FOLDER="rotators/exp_aug28_est_init_seed_$RANDOM_SEED"
CUDA_VISIBLE_DEVICES=1 python train.py train.random_seed=$RANDOM_SEED train.exps_folder=$EXP_FOLDER train.feats_backbone=clip train.model=rotator train.max_epochs=50 train.aggregator.type=two_random_index train.rotator.pretrained_cls=$PRETRAINED_CLS train.lr=5e-5 train.rotator.estimate_final_state=False &
EXP_FOLDER="rotators/exp_aug28_est_final_seed_$RANDOM_SEED"
CUDA_VISIBLE_DEVICES=2 python train.py train.random_seed=$RANDOM_SEED train.exps_folder=$EXP_FOLDER train.feats_backbone=clip train.model=rotator train.max_epochs=50 train.aggregator.type=two_random_index train.rotator.pretrained_cls=$PRETRAINED_CLS train.lr=5e-5 train.rotator.estimate_init_state=False &

## Seed 1
RANDOM_SEED=1
PRETRAINED_CLS="'/home/jdtho/snare/clips/exp_aug23_seed_1/clip-single_cls-two_random_index/checkpoints/epoch=0030-val_acc=0.81218.ckpt'"
EXP_FOLDER="rotators/exp_aug28_est_init_final_seed_$RANDOM_SEED"
CUDA_VISIBLE_DEVICES=0 python train.py train.random_seed=$RANDOM_SEED train.exps_folder=$EXP_FOLDER train.feats_backbone=clip train.model=rotator train.max_epochs=50 train.aggregator.type=two_random_index train.rotator.pretrained_cls=$PRETRAINED_CLS train.lr=5e-5 &
EXP_FOLDER="rotators/exp_aug28_est_init_seed_$RANDOM_SEED"
CUDA_VISIBLE_DEVICES=1 python train.py train.random_seed=$RANDOM_SEED train.exps_folder=$EXP_FOLDER train.feats_backbone=clip train.model=rotator train.max_epochs=50 train.aggregator.type=two_random_index train.rotator.pretrained_cls=$PRETRAINED_CLS train.lr=5e-5 train.rotator.estimate_final_state=False &
EXP_FOLDER="rotators/exp_aug28_est_final_seed_$RANDOM_SEED"
CUDA_VISIBLE_DEVICES=2 python train.py train.random_seed=$RANDOM_SEED train.exps_folder=$EXP_FOLDER train.feats_backbone=clip train.model=rotator train.max_epochs=50 train.aggregator.type=two_random_index train.rotator.pretrained_cls=$PRETRAINED_CLS train.lr=5e-5 train.rotator.estimate_init_state=False &

## Seed 2
RANDOM_SEED=2
PRETRAINED_CLS="'/home/jdtho/snare/clips/exp_aug23_seed_2/clip-single_cls-two_random_index/checkpoints/epoch=0018-val_acc=0.82284.ckpt'"
EXP_FOLDER="rotators/exp_aug28_est_init_final_seed_$RANDOM_SEED"
CUDA_VISIBLE_DEVICES=0 python train.py train.random_seed=$RANDOM_SEED train.exps_folder=$EXP_FOLDER train.feats_backbone=clip train.model=rotator train.max_epochs=50 train.aggregator.type=two_random_index train.rotator.pretrained_cls=$PRETRAINED_CLS train.lr=5e-5 &
EXP_FOLDER="rotators/exp_aug28_est_init_seed_$RANDOM_SEED"
CUDA_VISIBLE_DEVICES=1 python train.py train.random_seed=$RANDOM_SEED train.exps_folder=$EXP_FOLDER train.feats_backbone=clip train.model=rotator train.max_epochs=50 train.aggregator.type=two_random_index train.rotator.pretrained_cls=$PRETRAINED_CLS train.lr=5e-5 train.rotator.estimate_final_state=False &
EXP_FOLDER="rotators/exp_aug28_est_final_seed_$RANDOM_SEED"
CUDA_VISIBLE_DEVICES=2 python train.py train.random_seed=$RANDOM_SEED train.exps_folder=$EXP_FOLDER train.feats_backbone=clip train.model=rotator train.max_epochs=50 train.aggregator.type=two_random_index train.rotator.pretrained_cls=$PRETRAINED_CLS train.lr=5e-5 train.rotator.estimate_init_state=False &

## Seed 3
RANDOM_SEED=3
PRETRAINED_CLS="'/home/jdtho/snare/clips/exp_aug23_seed_3/clip-single_cls-two_random_index/checkpoints/epoch=0032-val_acc=0.82273.ckpt'"
EXP_FOLDER="rotators/exp_aug28_est_init_final_seed_$RANDOM_SEED"
CUDA_VISIBLE_DEVICES=0 python train.py train.random_seed=$RANDOM_SEED train.exps_folder=$EXP_FOLDER train.feats_backbone=clip train.model=rotator train.max_epochs=50 train.aggregator.type=two_random_index train.rotator.pretrained_cls=$PRETRAINED_CLS train.lr=5e-5 &
EXP_FOLDER="rotators/exp_aug28_est_init_seed_$RANDOM_SEED"
CUDA_VISIBLE_DEVICES=1 python train.py train.random_seed=$RANDOM_SEED train.exps_folder=$EXP_FOLDER train.feats_backbone=clip train.model=rotator train.max_epochs=50 train.aggregator.type=two_random_index train.rotator.pretrained_cls=$PRETRAINED_CLS train.lr=5e-5 train.rotator.estimate_final_state=False &
EXP_FOLDER="rotators/exp_aug28_est_final_seed_$RANDOM_SEED"
CUDA_VISIBLE_DEVICES=2 python train.py train.random_seed=$RANDOM_SEED train.exps_folder=$EXP_FOLDER train.feats_backbone=clip train.model=rotator train.max_epochs=50 train.aggregator.type=two_random_index train.rotator.pretrained_cls=$PRETRAINED_CLS train.lr=5e-5 train.rotator.estimate_init_state=False &

## Seed 4
RANDOM_SEED=4
PRETRAINED_CLS="'/home/jdtho/snare/clips/exp_aug23_seed_4/clip-single_cls-two_random_index/checkpoints/epoch=0035-val_acc=0.81721.ckpt'"
EXP_FOLDER="rotators/exp_aug28_est_init_final_seed_$RANDOM_SEED"
CUDA_VISIBLE_DEVICES=0 python train.py train.random_seed=$RANDOM_SEED train.exps_folder=$EXP_FOLDER train.feats_backbone=clip train.model=rotator train.max_epochs=50 train.aggregator.type=two_random_index train.rotator.pretrained_cls=$PRETRAINED_CLS train.lr=5e-5 &
EXP_FOLDER="rotators/exp_aug28_est_init_seed_$RANDOM_SEED"
CUDA_VISIBLE_DEVICES=1 python train.py train.random_seed=$RANDOM_SEED train.exps_folder=$EXP_FOLDER train.feats_backbone=clip train.model=rotator train.max_epochs=50 train.aggregator.type=two_random_index train.rotator.pretrained_cls=$PRETRAINED_CLS train.lr=5e-5 train.rotator.estimate_final_state=False &
EXP_FOLDER="rotators/exp_aug28_est_final_seed_$RANDOM_SEED"
CUDA_VISIBLE_DEVICES=2 python train.py train.random_seed=$RANDOM_SEED train.exps_folder=$EXP_FOLDER train.feats_backbone=clip train.model=rotator train.max_epochs=50 train.aggregator.type=two_random_index train.rotator.pretrained_cls=$PRETRAINED_CLS train.lr=5e-5 train.rotator.estimate_init_state=False &

## Seed 5
RANDOM_SEED=5
PRETRAINED_CLS="'/home/jdtho/snare/clips/exp_aug23_seed_5/clip-single_cls-two_random_index/checkpoints/epoch=0017-val_acc=0.81861.ckpt'"
EXP_FOLDER="rotators/exp_aug28_est_init_final_seed_$RANDOM_SEED"
CUDA_VISIBLE_DEVICES=0 python train.py train.random_seed=$RANDOM_SEED train.exps_folder=$EXP_FOLDER train.feats_backbone=clip train.model=rotator train.max_epochs=50 train.aggregator.type=two_random_index train.rotator.pretrained_cls=$PRETRAINED_CLS train.lr=5e-5 &
EXP_FOLDER="rotators/exp_aug28_est_init_seed_$RANDOM_SEED"
CUDA_VISIBLE_DEVICES=1 python train.py train.random_seed=$RANDOM_SEED train.exps_folder=$EXP_FOLDER train.feats_backbone=clip train.model=rotator train.max_epochs=50 train.aggregator.type=two_random_index train.rotator.pretrained_cls=$PRETRAINED_CLS train.lr=5e-5 train.rotator.estimate_final_state=False &
EXP_FOLDER="rotators/exp_aug28_est_final_seed_$RANDOM_SEED"
CUDA_VISIBLE_DEVICES=2 python train.py train.random_seed=$RANDOM_SEED train.exps_folder=$EXP_FOLDER train.feats_backbone=clip train.model=rotator train.max_epochs=50 train.aggregator.type=two_random_index train.rotator.pretrained_cls=$PRETRAINED_CLS train.lr=5e-5 train.rotator.estimate_init_state=False &

## Seed 6
RANDOM_SEED=6
PRETRAINED_CLS="'/home/jdtho/snare/clips/exp_aug23_seed_6/clip-single_cls-two_random_index/checkpoints/epoch=0014-val_acc=0.82397.ckpt'"
EXP_FOLDER="rotators/exp_aug28_est_init_final_seed_$RANDOM_SEED"
CUDA_VISIBLE_DEVICES=0 python train.py train.random_seed=$RANDOM_SEED train.exps_folder=$EXP_FOLDER train.feats_backbone=clip train.model=rotator train.max_epochs=50 train.aggregator.type=two_random_index train.rotator.pretrained_cls=$PRETRAINED_CLS train.lr=5e-5 &
EXP_FOLDER="rotators/exp_aug28_est_init_seed_$RANDOM_SEED"
CUDA_VISIBLE_DEVICES=1 python train.py train.random_seed=$RANDOM_SEED train.exps_folder=$EXP_FOLDER train.feats_backbone=clip train.model=rotator train.max_epochs=50 train.aggregator.type=two_random_index train.rotator.pretrained_cls=$PRETRAINED_CLS train.lr=5e-5 train.rotator.estimate_final_state=False &
EXP_FOLDER="rotators/exp_aug28_est_final_seed_$RANDOM_SEED"
CUDA_VISIBLE_DEVICES=2 python train.py train.random_seed=$RANDOM_SEED train.exps_folder=$EXP_FOLDER train.feats_backbone=clip train.model=rotator train.max_epochs=50 train.aggregator.type=two_random_index train.rotator.pretrained_cls=$PRETRAINED_CLS train.lr=5e-5 train.rotator.estimate_init_state=False &

## Seed 7
# RANDOM_SEED=7
PRETRAINED_CLS="'/home/jdtho/snare/clips/exp_aug23_seed_7/clip-single_cls-two_random_index/checkpoints/epoch=0021-val_acc=0.81423.ckpt'"
EXP_FOLDER="rotators/exp_aug28_est_init_final_seed_$RANDOM_SEED"
CUDA_VISIBLE_DEVICES=0 python train.py train.random_seed=$RANDOM_SEED train.exps_folder=$EXP_FOLDER train.feats_backbone=clip train.model=rotator train.max_epochs=50 train.aggregator.type=two_random_index train.rotator.pretrained_cls=$PRETRAINED_CLS train.lr=5e-5 &
EXP_FOLDER="rotators/exp_aug28_est_init_seed_$RANDOM_SEED"
CUDA_VISIBLE_DEVICES=1 python train.py train.random_seed=$RANDOM_SEED train.exps_folder=$EXP_FOLDER train.feats_backbone=clip train.model=rotator train.max_epochs=50 train.aggregator.type=two_random_index train.rotator.pretrained_cls=$PRETRAINED_CLS train.lr=5e-5 train.rotator.estimate_final_state=False &
EXP_FOLDER="rotators/exp_aug28_est_final_seed_$RANDOM_SEED"
CUDA_VISIBLE_DEVICES=2 python train.py train.random_seed=$RANDOM_SEED train.exps_folder=$EXP_FOLDER train.feats_backbone=clip train.model=rotator train.max_epochs=50 train.aggregator.type=two_random_index train.rotator.pretrained_cls=$PRETRAINED_CLS train.lr=5e-5 train.rotator.estimate_init_state=False &

## Seed 8
RANDOM_SEED=8
PRETRAINED_CLS="'/home/jdtho/snare/clips/exp_aug23_seed_8/clip-single_cls-two_random_index/checkpoints/epoch=0027-val_acc=0.81742.ckpt'"
EXP_FOLDER="rotators/exp_aug28_est_init_final_seed_$RANDOM_SEED"
CUDA_VISIBLE_DEVICES=0 python train.py train.random_seed=$RANDOM_SEED train.exps_folder=$EXP_FOLDER train.feats_backbone=clip train.model=rotator train.max_epochs=50 train.aggregator.type=two_random_index train.rotator.pretrained_cls=$PRETRAINED_CLS train.lr=5e-5 &
EXP_FOLDER="rotators/exp_aug28_est_init_seed_$RANDOM_SEED"
CUDA_VISIBLE_DEVICES=1 python train.py train.random_seed=$RANDOM_SEED train.exps_folder=$EXP_FOLDER train.feats_backbone=clip train.model=rotator train.max_epochs=50 train.aggregator.type=two_random_index train.rotator.pretrained_cls=$PRETRAINED_CLS train.lr=5e-5 train.rotator.estimate_final_state=False &
EXP_FOLDER="rotators/exp_aug28_est_final_seed_$RANDOM_SEED"
CUDA_VISIBLE_DEVICES=2 python train.py train.random_seed=$RANDOM_SEED train.exps_folder=$EXP_FOLDER train.feats_backbone=clip train.model=rotator train.max_epochs=50 train.aggregator.type=two_random_index train.rotator.pretrained_cls=$PRETRAINED_CLS train.lr=5e-5 train.rotator.estimate_init_state=False &

## Seed 9
RANDOM_SEED=9
PRETRAINED_CLS="'/home/jdtho/snare/clips/exp_aug23_seed_9/clip-single_cls-two_random_index/checkpoints/epoch=0014-val_acc=0.81726.ckpt'"
EXP_FOLDER="rotators/exp_aug28_est_init_final_seed_$RANDOM_SEED"
CUDA_VISIBLE_DEVICES=0 python train.py train.random_seed=$RANDOM_SEED train.exps_folder=$EXP_FOLDER train.feats_backbone=clip train.model=rotator train.max_epochs=50 train.aggregator.type=two_random_index train.rotator.pretrained_cls=$PRETRAINED_CLS train.lr=5e-5 &
EXP_FOLDER="rotators/exp_aug28_est_init_seed_$RANDOM_SEED"
CUDA_VISIBLE_DEVICES=1 python train.py train.random_seed=$RANDOM_SEED train.exps_folder=$EXP_FOLDER train.feats_backbone=clip train.model=rotator train.max_epochs=50 train.aggregator.type=two_random_index train.rotator.pretrained_cls=$PRETRAINED_CLS train.lr=5e-5 train.rotator.estimate_final_state=False &
EXP_FOLDER="rotators/exp_aug28_est_final_seed_$RANDOM_SEED"
CUDA_VISIBLE_DEVICES=2 python train.py train.random_seed=$RANDOM_SEED train.exps_folder=$EXP_FOLDER train.feats_backbone=clip train.model=rotator train.max_epochs=50 train.aggregator.type=two_random_index train.rotator.pretrained_cls=$PRETRAINED_CLS train.lr=5e-5 train.rotator.estimate_init_state=False &
