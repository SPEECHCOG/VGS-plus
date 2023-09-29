#!/bin/sh
export PATH="/projappl/project_2001315/khazar/con/vf/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3

data_root=$1
fb_w2v2_weights_fn="../../../../model/wav2vec_small.pt"
exp_dir="../../expS1/"
libri_fn_root="../../../../datavf/ssl6M_root/"
twd="../../twd/"

python \
../run_spokencoco.py \
--subset "subset1" \
--data_root ${data_root} \
--trained_weights_dir ${twd} \
--fb_w2v2_weights_fn ${fb_w2v2_weights_fn} \
--exp_dir ${exp_dir} \
--libri_fn_root ${libri_fn_root} \
--batch_size 64 \
--val_batch_size 64 \
--val_cross_batch_size 64 \
--n_epochs 200 \
--n_print_steps 20 \
--n_val_steps 1000 \
--lr 0.0001 \
--warmup_fraction 0.1 \
--normalize \
--xtrm_layers 1 \
--trm_layers 6 \
--fine_matching_weight 0.0 \
--coarse_matching_weight 1.0 \
--libri_w2v2_weight 0.0 \
--caption_w2v2_weight 1.0 \
--feature_grad_mult 1.0 \
--trim_mask \
--layer_use 7 \

