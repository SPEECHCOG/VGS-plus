#!/bin/sh
source activate fastvgs
export CUDA_VISIBLE_DEVICES=0

python \
../cuDNN-test.py
