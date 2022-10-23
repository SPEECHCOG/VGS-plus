#!/bin/sh
source activate fastvgs

python tester_ABX.py

conda activate zerospeech2021
zerospeech2021-evaluate /worktmp/khorrami/current/ZeroSpeech/data/  /worktmp/khorrami/current/ZeroSpeech/submission/ -o /worktmp/khorrami/current/ZeroSpeech/output/  -j12 --no-lexical --no-syntactic --no-semantic
rm -r /worktmp/khorrami/current/ZeroSpeech/submission/phonetic/dev-clean
