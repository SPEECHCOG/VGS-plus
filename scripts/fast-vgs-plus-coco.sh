#!/bin/sh
source activate fastvgs
export CUDA_VISIBLE_DEVICES=0,1,2,3

data_root=$1
raw_audio_base_path=$2
fb_w2v2_weights_fn=$3
exp_dir=$4
libri_fn_root=$5

python \
../run_spokencoco.py \
--data_root ${data_root} \
--raw_audio_base_path ${raw_audio_base_path} \
--fb_w2v2_weights_fn ${fb_w2v2_weights_fn} \
--exp_dir ${exp_dir} \
--libri_fn_root ${libri_fn_root} \
--batch_size 32 \
--val_batch_size 32 \
--val_cross_batch_size 8 \
--n_epochs 20 \
--n_print_steps 100 \
--n_val_steps 2000 \
--lr 0.0001 \
--warmup_fraction 0.1 \
--normalize \
--xtrm_layers 1 \
--trm_layers 2 \
--fine_matching_weight 0.0 \
--coarse_matching_weight 1 \
--libri_w2v2_weight 0.0 \
--caption_w2v2_weight 1.0 \
--coarse_to_fine_retrieve \
--feature_grad_mult 0.1 \
--layer_use 7

