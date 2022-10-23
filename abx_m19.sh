#!/bin/sh
source activate fastvgs

python abx_m19.py --mytarget_layer 2 --mytwd '/worktmp/khorrami/current/FaST/experiments/model19base1/185040_bundle.pth'
conda activate zerospeech2021
mkdir /worktmp/khorrami/current/ZeroSpeech/output/m19base1E20L2
zerospeech2021-evaluate /worktmp/khorrami/current/ZeroSpeech/data/  /worktmp/khorrami/current/ZeroSpeech/submission/ -o /worktmp/khorrami/current/ZeroSpeech/output/m19base1E20L2  -j12 --no-lexical --no-syntactic --no-semantic --force-cpu
rm -r /worktmp/khorrami/current/ZeroSpeech/submission/phonetic/dev-clean

conda activate fastvgs
python abx_m19.py --mytarget_layer 3 --mytwd '/worktmp/khorrami/current/FaST/experiments/model19base1/185040_bundle.pth'
conda activate zerospeech2021
mkdir /worktmp/khorrami/current/ZeroSpeech/output/m19base1E20L3
zerospeech2021-evaluate /worktmp/khorrami/current/ZeroSpeech/data/  /worktmp/khorrami/current/ZeroSpeech/submission/ -o /worktmp/khorrami/current/ZeroSpeech/output/m19base1E20L3  -j12 --no-lexical --no-syntactic --no-semantic --force-cpu
rm -r /worktmp/khorrami/current/ZeroSpeech/submission/phonetic/dev-clean

conda activate fastvgs
python abx_m19.py --mytarget_layer 4 --mytwd '/worktmp/khorrami/current/FaST/experiments/model19base1/185040_bundle.pth'
conda activate zerospeech2021
mkdir /worktmp/khorrami/current/ZeroSpeech/output/m19base1E20L4
zerospeech2021-evaluate /worktmp/khorrami/current/ZeroSpeech/data/  /worktmp/khorrami/current/ZeroSpeech/submission/ -o /worktmp/khorrami/current/ZeroSpeech/output/m19base1E20L4  -j12 --no-lexical --no-syntactic --no-semantic --force-cpu
rm -r /worktmp/khorrami/current/ZeroSpeech/submission/phonetic/dev-clean

conda activate fastvgs
python abx_m19.py --mytarget_layer 5 --mytwd '/worktmp/khorrami/current/FaST/experiments/model19base1/185040_bundle.pth'
conda activate zerospeech2021
mkdir /worktmp/khorrami/current/ZeroSpeech/output/m19base1E20L5
zerospeech2021-evaluate /worktmp/khorrami/current/ZeroSpeech/data/  /worktmp/khorrami/current/ZeroSpeech/submission/ -o /worktmp/khorrami/current/ZeroSpeech/output/m19base1E20L5  -j12 --no-lexical --no-syntactic --no-semantic --force-cpu
rm -r /worktmp/khorrami/current/ZeroSpeech/submission/phonetic/dev-clean


