#!/bin/sh
source activate fastvgs

python abx_19base3_E1L3.py

mkdir /worktmp/khorrami/current/ZeroSpeech/output/abx_19base3_E1L3

conda activate zerospeech2021
zerospeech2021-evaluate /worktmp/khorrami/current/ZeroSpeech/data/  /worktmp/khorrami/current/ZeroSpeech/submission/ -o /worktmp/khorrami/current/ZeroSpeech/output/abx_19base3_E1L3  -j12 --no-lexical --no-syntactic --no-semantic
rm -r /worktmp/khorrami/current/ZeroSpeech/submission/phonetic/dev-clean
