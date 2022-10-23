#!/bin/sh
source activate fastvgs

python abx_m19.py --mytarget_layer 1 --mytwd '/worktmp/khorrami/current/FaST/experiments/model19base1/9252_bundle.pth'
conda activate zerospeech2021
mkdir m19base1E1L1
zerospeech2021-evaluate /worktmp/khorrami/current/ZeroSpeech/data/  /worktmp/khorrami/current/ZeroSpeech/submission/ -o /worktmp/khorrami/current/ZeroSpeech/output/m19base1E1L1  -j12 --no-lexical --no-syntactic --no-semantic --force-cpu
rm -r /worktmp/khorrami/current/ZeroSpeech/submission/phonetic/dev-clean

python abx_m19.py --mytarget_layer 2 --mytwd '/worktmp/khorrami/current/FaST/experiments/model19base1/9252_bundle.pth'
conda activate zerospeech2021
mkdir m19base1E1L2
zerospeech2021-evaluate /worktmp/khorrami/current/ZeroSpeech/data/  /worktmp/khorrami/current/ZeroSpeech/submission/ -o /worktmp/khorrami/current/ZeroSpeech/output/m19base1E1L2  -j12 --no-lexical --no-syntactic --no-semantic --force-cpu
rm -r /worktmp/khorrami/current/ZeroSpeech/submission/phonetic/dev-clean


