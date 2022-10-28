#!/bin/sh
source activate fastvgs
python abx_m20.py --mytarget_layer 1 --mytwd '/worktmp/khorrami/current/FaST/experiments/model20base3/9252_bundle.pth'
conda activate zerospeech2021
OUTFILE="/worktmp/khorrami/current/ZeroSpeech/output/m20base3E1L1"
mkdir $OUTFILE
zerospeech2021-evaluate /worktmp/khorrami/current/ZeroSpeech/data/  /worktmp/khorrami/current/ZeroSpeech/submission/ -o $OUTFILE  -j12 --no-lexical --no-syntactic --no-semantic --force-cpu
rm -r /worktmp/khorrami/current/ZeroSpeech/submission/phonetic/dev-clean

conda activate fastvgs
python abx_m20.py --mytarget_layer 2 --mytwd '/worktmp/khorrami/current/FaST/experiments/model20base3/9252_bundle.pth'
conda activate zerospeech2021
OUTFILE="/worktmp/khorrami/current/ZeroSpeech/output/m20base3E1L2"
mkdir $OUTFILE
zerospeech2021-evaluate /worktmp/khorrami/current/ZeroSpeech/data/  /worktmp/khorrami/current/ZeroSpeech/submission/ -o $OUTFILE  -j12 --no-lexical --no-syntactic --no-semantic --force-cpu
rm -r /worktmp/khorrami/current/ZeroSpeech/submission/phonetic/dev-clean

conda activate fastvgs
python abx_m20.py --mytarget_layer 3 --mytwd '/worktmp/khorrami/current/FaST/experiments/model20base3/9252_bundle.pth'
conda activate zerospeech2021
OUTFILE="/worktmp/khorrami/current/ZeroSpeech/output/m20base3E1L3"
mkdir $OUTFILE
zerospeech2021-evaluate /worktmp/khorrami/current/ZeroSpeech/data/  /worktmp/khorrami/current/ZeroSpeech/submission/ -o $OUTFILE  -j12 --no-lexical --no-syntactic --no-semantic --force-cpu
rm -r /worktmp/khorrami/current/ZeroSpeech/submission/phonetic/dev-clean

conda activate fastvgs
python abx_m20.py --mytarget_layer 4 --mytwd '/worktmp/khorrami/current/FaST/experiments/model20base3/9252_bundle.pth'
conda activate zerospeech2021
OUTFILE="/worktmp/khorrami/current/ZeroSpeech/output/m20base3E1L4"
mkdir $OUTFILE
zerospeech2021-evaluate /worktmp/khorrami/current/ZeroSpeech/data/  /worktmp/khorrami/current/ZeroSpeech/submission/ -o $OUTFILE  -j12 --no-lexical --no-syntactic --no-semantic --force-cpu
rm -r /worktmp/khorrami/current/ZeroSpeech/submission/phonetic/dev-clean

conda activate fastvgs
python abx_m20.py --mytarget_layer 5 --mytwd '/worktmp/khorrami/current/FaST/experiments/model20base3/9252_bundle.pth'
conda activate zerospeech2021
OUTFILE="/worktmp/khorrami/current/ZeroSpeech/output/m20base3E1L5"
mkdir $OUTFILE
zerospeech2021-evaluate /worktmp/khorrami/current/ZeroSpeech/data/  /worktmp/khorrami/current/ZeroSpeech/submission/ -o $OUTFILE  -j12 --no-lexical --no-syntactic --no-semantic --force-cpu
rm -r /worktmp/khorrami/current/ZeroSpeech/submission/phonetic/dev-clean


conda activate fastvgs
python abx_m20.py --mytarget_layer 1 --mytwd '/worktmp/khorrami/current/FaST/experiments/model20base3/46260_bundle.pth'
conda activate zerospeech2021
OUTFILE="/worktmp/khorrami/current/ZeroSpeech/output/m20base3E5L1"
mkdir $OUTFILE
zerospeech2021-evaluate /worktmp/khorrami/current/ZeroSpeech/data/  /worktmp/khorrami/current/ZeroSpeech/submission/ -o $OUTFILE  -j12 --no-lexical --no-syntactic --no-semantic --force-cpu
rm -r /worktmp/khorrami/current/ZeroSpeech/submission/phonetic/dev-clean

conda activate fastvgs
python abx_m20.py --mytarget_layer 2 --mytwd '/worktmp/khorrami/current/FaST/experiments/model20base3/46260_bundle.pth'
conda activate zerospeech2021
OUTFILE="/worktmp/khorrami/current/ZeroSpeech/output/m20base3E5L2"
mkdir $OUTFILE
zerospeech2021-evaluate /worktmp/khorrami/current/ZeroSpeech/data/  /worktmp/khorrami/current/ZeroSpeech/submission/ -o $OUTFILE  -j12 --no-lexical --no-syntactic --no-semantic --force-cpu
rm -r /worktmp/khorrami/current/ZeroSpeech/submission/phonetic/dev-clean

conda activate fastvgs
python abx_m20.py --mytarget_layer 3 --mytwd '/worktmp/khorrami/current/FaST/experiments/model20base3/46260_bundle.pth'
conda activate zerospeech2021
OUTFILE="/worktmp/khorrami/current/ZeroSpeech/output/m20base3E5L3"
mkdir $OUTFILE
zerospeech2021-evaluate /worktmp/khorrami/current/ZeroSpeech/data/  /worktmp/khorrami/current/ZeroSpeech/submission/ -o $OUTFILE  -j12 --no-lexical --no-syntactic --no-semantic --force-cpu
rm -r /worktmp/khorrami/current/ZeroSpeech/submission/phonetic/dev-clean

conda activate fastvgs
python abx_m20.py --mytarget_layer 4 --mytwd '/worktmp/khorrami/current/FaST/experiments/model20base3/46260_bundle.pth'
conda activate zerospeech2021
OUTFILE="/worktmp/khorrami/current/ZeroSpeech/output/m20base3E5L4"
mkdir $OUTFILE
zerospeech2021-evaluate /worktmp/khorrami/current/ZeroSpeech/data/  /worktmp/khorrami/current/ZeroSpeech/submission/ -o $OUTFILE  -j12 --no-lexical --no-syntactic --no-semantic --force-cpu
rm -r /worktmp/khorrami/current/ZeroSpeech/submission/phonetic/dev-clean

conda activate fastvgs
python abx_m20.py --mytarget_layer 5 --mytwd '/worktmp/khorrami/current/FaST/experiments/model20base3/46260_bundle.pth'
conda activate zerospeech2021
OUTFILE="/worktmp/khorrami/current/ZeroSpeech/output/m20base3E5L5"
mkdir $OUTFILE
zerospeech2021-evaluate /worktmp/khorrami/current/ZeroSpeech/data/  /worktmp/khorrami/current/ZeroSpeech/submission/ -o $OUTFILE  -j12 --no-lexical --no-syntactic --no-semantic --force-cpu
rm -r /worktmp/khorrami/current/ZeroSpeech/submission/phonetic/dev-clean

conda activate fastvgs
python abx_m20.py --mytarget_layer 1 --mytwd '/worktmp/khorrami/current/FaST/experiments/model20base3/92520_bundle.pth'
conda activate zerospeech2021
OUTFILE="/worktmp/khorrami/current/ZeroSpeech/output/m20base3E10L1"
mkdir $OUTFILE
zerospeech2021-evaluate /worktmp/khorrami/current/ZeroSpeech/data/  /worktmp/khorrami/current/ZeroSpeech/submission/ -o $OUTFILE  -j12 --no-lexical --no-syntactic --no-semantic --force-cpu
rm -r /worktmp/khorrami/current/ZeroSpeech/submission/phonetic/dev-clean

conda activate fastvgs
python abx_m20.py --mytarget_layer 2 --mytwd '/worktmp/khorrami/current/FaST/experiments/model20base3/92520_bundle.pth'
conda activate zerospeech2021
OUTFILE="/worktmp/khorrami/current/ZeroSpeech/output/m20base3E10L2"
mkdir $OUTFILE
zerospeech2021-evaluate /worktmp/khorrami/current/ZeroSpeech/data/  /worktmp/khorrami/current/ZeroSpeech/submission/ -o $OUTFILE  -j12 --no-lexical --no-syntactic --no-semantic --force-cpu
rm -r /worktmp/khorrami/current/ZeroSpeech/submission/phonetic/dev-clean

conda activate fastvgs
python abx_m20.py --mytarget_layer 3 --mytwd '/worktmp/khorrami/current/FaST/experiments/model20base3/92520_bundle.pth'
conda activate zerospeech2021
OUTFILE="/worktmp/khorrami/current/ZeroSpeech/output/m20base3E10L3"
mkdir $OUTFILE
zerospeech2021-evaluate /worktmp/khorrami/current/ZeroSpeech/data/  /worktmp/khorrami/current/ZeroSpeech/submission/ -o $OUTFILE  -j12 --no-lexical --no-syntactic --no-semantic --force-cpu
rm -r /worktmp/khorrami/current/ZeroSpeech/submission/phonetic/dev-clean

conda activate fastvgs
python abx_m20.py --mytarget_layer 4 --mytwd '/worktmp/khorrami/current/FaST/experiments/model20base3/92520_bundle.pth'
conda activate zerospeech2021
OUTFILE="/worktmp/khorrami/current/ZeroSpeech/output/m20base3E10L4"
mkdir $OUTFILE
zerospeech2021-evaluate /worktmp/khorrami/current/ZeroSpeech/data/  /worktmp/khorrami/current/ZeroSpeech/submission/ -o $OUTFILE  -j12 --no-lexical --no-syntactic --no-semantic --force-cpu
rm -r /worktmp/khorrami/current/ZeroSpeech/submission/phonetic/dev-clean

conda activate fastvgs
python abx_m20.py --mytarget_layer 5 --mytwd '/worktmp/khorrami/current/FaST/experiments/model20base3/92520_bundle.pth'
conda activate zerospeech2021
OUTFILE="/worktmp/khorrami/current/ZeroSpeech/output/m20base3E10L5"
mkdir $OUTFILE
zerospeech2021-evaluate /worktmp/khorrami/current/ZeroSpeech/data/  /worktmp/khorrami/current/ZeroSpeech/submission/ -o $OUTFILE  -j12 --no-lexical --no-syntactic --no-semantic --force-cpu
rm -r /worktmp/khorrami/current/ZeroSpeech/submission/phonetic/dev-clean

conda activate fastvgs
python abx_m20.py --mytarget_layer 1 --mytwd '/worktmp/khorrami/current/FaST/experiments/model20base3/138780_bundle.pth'
conda activate zerospeech2021
OUTFILE="/worktmp/khorrami/current/ZeroSpeech/output/m20base3E15L1"
mkdir $OUTFILE
zerospeech2021-evaluate /worktmp/khorrami/current/ZeroSpeech/data/  /worktmp/khorrami/current/ZeroSpeech/submission/ -o $OUTFILE  -j12 --no-lexical --no-syntactic --no-semantic --force-cpu
rm -r /worktmp/khorrami/current/ZeroSpeech/submission/phonetic/dev-clean

conda activate fastvgs
python abx_m20.py --mytarget_layer 2 --mytwd '/worktmp/khorrami/current/FaST/experiments/model20base3/138780_bundle.pth'
conda activate zerospeech2021
OUTFILE="/worktmp/khorrami/current/ZeroSpeech/output/m20base3E15L2"
mkdir $OUTFILE
zerospeech2021-evaluate /worktmp/khorrami/current/ZeroSpeech/data/  /worktmp/khorrami/current/ZeroSpeech/submission/ -o $OUTFILE  -j12 --no-lexical --no-syntactic --no-semantic --force-cpu
rm -r /worktmp/khorrami/current/ZeroSpeech/submission/phonetic/dev-clean

conda activate fastvgs
python abx_m20.py --mytarget_layer 3 --mytwd '/worktmp/khorrami/current/FaST/experiments/model20base3/138780_bundle.pth'
conda activate zerospeech2021
OUTFILE="/worktmp/khorrami/current/ZeroSpeech/output/m20base3E15L3"
mkdir $OUTFILE
zerospeech2021-evaluate /worktmp/khorrami/current/ZeroSpeech/data/  /worktmp/khorrami/current/ZeroSpeech/submission/ -o $OUTFILE  -j12 --no-lexical --no-syntactic --no-semantic --force-cpu
rm -r /worktmp/khorrami/current/ZeroSpeech/submission/phonetic/dev-clean

conda activate fastvgs
python abx_m20.py --mytarget_layer 4 --mytwd '/worktmp/khorrami/current/FaST/experiments/model20base3/138780_bundle.pth'
conda activate zerospeech2021
OUTFILE="/worktmp/khorrami/current/ZeroSpeech/output/m20base3E15L4"
mkdir $OUTFILE
zerospeech2021-evaluate /worktmp/khorrami/current/ZeroSpeech/data/  /worktmp/khorrami/current/ZeroSpeech/submission/ -o $OUTFILE  -j12 --no-lexical --no-syntactic --no-semantic --force-cpu
rm -r /worktmp/khorrami/current/ZeroSpeech/submission/phonetic/dev-clean

conda activate fastvgs
python abx_m20.py --mytarget_layer 5 --mytwd '/worktmp/khorrami/current/FaST/experiments/model20base3/138780_bundle.pth'
conda activate zerospeech2021
OUTFILE="/worktmp/khorrami/current/ZeroSpeech/output/m20base3E15L5"
mkdir $OUTFILE
zerospeech2021-evaluate /worktmp/khorrami/current/ZeroSpeech/data/  /worktmp/khorrami/current/ZeroSpeech/submission/ -o $OUTFILE  -j12 --no-lexical --no-syntactic --no-semantic --force-cpu
rm -r /worktmp/khorrami/current/ZeroSpeech/submission/phonetic/dev-clean


conda activate fastvgs
python abx_m20.py --mytarget_layer 1 --mytwd '/worktmp/khorrami/current/FaST/experiments/model20base3/185040_bundle.pth'
conda activate zerospeech2021
OUTFILE="/worktmp/khorrami/current/ZeroSpeech/output/m20base3E20L1"
mkdir $OUTFILE
zerospeech2021-evaluate /worktmp/khorrami/current/ZeroSpeech/data/  /worktmp/khorrami/current/ZeroSpeech/submission/ -o $OUTFILE  -j12 --no-lexical --no-syntactic --no-semantic --force-cpu
rm -r /worktmp/khorrami/current/ZeroSpeech/submission/phonetic/dev-clean

conda activate fastvgs
python abx_m20.py --mytarget_layer 2 --mytwd '/worktmp/khorrami/current/FaST/experiments/model20base3/185040_bundle.pth'
conda activate zerospeech2021
OUTFILE="/worktmp/khorrami/current/ZeroSpeech/output/m20base3E20L2"
mkdir $OUTFILE
zerospeech2021-evaluate /worktmp/khorrami/current/ZeroSpeech/data/  /worktmp/khorrami/current/ZeroSpeech/submission/ -o $OUTFILE  -j12 --no-lexical --no-syntactic --no-semantic --force-cpu
rm -r /worktmp/khorrami/current/ZeroSpeech/submission/phonetic/dev-clean

conda activate fastvgs
python abx_m20.py --mytarget_layer 3 --mytwd '/worktmp/khorrami/current/FaST/experiments/model20base3/185040_bundle.pth'
conda activate zerospeech2021
OUTFILE="/worktmp/khorrami/current/ZeroSpeech/output/m20base3E20L3"
mkdir $OUTFILE
zerospeech2021-evaluate /worktmp/khorrami/current/ZeroSpeech/data/  /worktmp/khorrami/current/ZeroSpeech/submission/ -o $OUTFILE  -j12 --no-lexical --no-syntactic --no-semantic --force-cpu
rm -r /worktmp/khorrami/current/ZeroSpeech/submission/phonetic/dev-clean

conda activate fastvgs
python abx_m20.py --mytarget_layer 4 --mytwd '/worktmp/khorrami/current/FaST/experiments/model20base3/185040_bundle.pth'
conda activate zerospeech2021
OUTFILE="/worktmp/khorrami/current/ZeroSpeech/output/m20base3E20L4"
mkdir $OUTFILE
zerospeech2021-evaluate /worktmp/khorrami/current/ZeroSpeech/data/  /worktmp/khorrami/current/ZeroSpeech/submission/ -o $OUTFILE  -j12 --no-lexical --no-syntactic --no-semantic --force-cpu
rm -r /worktmp/khorrami/current/ZeroSpeech/submission/phonetic/dev-clean

conda activate fastvgs
python abx_m20.py --mytarget_layer 5 --mytwd '/worktmp/khorrami/current/FaST/experiments/model20base3/185040_bundle.pth'
conda activate zerospeech2021
OUTFILE="/worktmp/khorrami/current/ZeroSpeech/output/m20base3E20L5"
mkdir $OUTFILE
zerospeech2021-evaluate /worktmp/khorrami/current/ZeroSpeech/data/  /worktmp/khorrami/current/ZeroSpeech/submission/ -o $OUTFILE  -j12 --no-lexical --no-syntactic --no-semantic --force-cpu
rm -r /worktmp/khorrami/current/ZeroSpeech/submission/phonetic/dev-clean

