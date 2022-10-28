#!/bin/sh
source activate fastvgs
python abx_m20.py --mytarget_layer 1 --mytwd '/worktmp/khorrami/current/FaST/experiments/model20base2/9252_bundle.pth'
conda activate zerospeech2021
OUTFILE="/worktmp/khorrami/current/ZeroSpeech/output/m20base2E1L1"
mkdir $OUTFILE
zerospeech2021-evaluate /worktmp/khorrami/current/ZeroSpeech/data/  /worktmp/khorrami/current/ZeroSpeech/submission/ -o $OUTFILE  -j12 --no-lexical --no-syntactic --no-semantic --force-cpu
rm -r /worktmp/khorrami/current/ZeroSpeech/submission/phonetic/dev-clean

conda activate fastvgs
python abx_m20.py --mytarget_layer 2 --mytwd '/worktmp/khorrami/current/FaST/experiments/model20base2/9252_bundle.pth'
conda activate zerospeech2021
OUTFILE="/worktmp/khorrami/current/ZeroSpeech/output/m20base2E1L2"
mkdir $OUTFILE
zerospeech2021-evaluate /worktmp/khorrami/current/ZeroSpeech/data/  /worktmp/khorrami/current/ZeroSpeech/submission/ -o $OUTFILE  -j12 --no-lexical --no-syntactic --no-semantic --force-cpu
rm -r /worktmp/khorrami/current/ZeroSpeech/submission/phonetic/dev-clean

conda activate fastvgs
python abx_m20.py --mytarget_layer 3 --mytwd '/worktmp/khorrami/current/FaST/experiments/model20base2/9252_bundle.pth'
conda activate zerospeech2021
OUTFILE="/worktmp/khorrami/current/ZeroSpeech/output/m20base2E1L3"
mkdir $OUTFILE
zerospeech2021-evaluate /worktmp/khorrami/current/ZeroSpeech/data/  /worktmp/khorrami/current/ZeroSpeech/submission/ -o $OUTFILE  -j12 --no-lexical --no-syntactic --no-semantic --force-cpu
rm -r /worktmp/khorrami/current/ZeroSpeech/submission/phonetic/dev-clean

conda activate fastvgs
python abx_m20.py --mytarget_layer 4 --mytwd '/worktmp/khorrami/current/FaST/experiments/model20base2/9252_bundle.pth'
conda activate zerospeech2021
OUTFILE="/worktmp/khorrami/current/ZeroSpeech/output/m20base2E1L4"
mkdir $OUTFILE
zerospeech2021-evaluate /worktmp/khorrami/current/ZeroSpeech/data/  /worktmp/khorrami/current/ZeroSpeech/submission/ -o $OUTFILE  -j12 --no-lexical --no-syntactic --no-semantic --force-cpu
rm -r /worktmp/khorrami/current/ZeroSpeech/submission/phonetic/dev-clean

conda activate fastvgs
python abx_m20.py --mytarget_layer 5 --mytwd '/worktmp/khorrami/current/FaST/experiments/model20base2/9252_bundle.pth'
conda activate zerospeech2021
OUTFILE="/worktmp/khorrami/current/ZeroSpeech/output/m20base2E1L5"
mkdir $OUTFILE
zerospeech2021-evaluate /worktmp/khorrami/current/ZeroSpeech/data/  /worktmp/khorrami/current/ZeroSpeech/submission/ -o $OUTFILE  -j12 --no-lexical --no-syntactic --no-semantic --force-cpu
rm -r /worktmp/khorrami/current/ZeroSpeech/submission/phonetic/dev-clean


conda activate fastvgs
python abx_m20.py --mytarget_layer 1 --mytwd '/worktmp/khorrami/current/FaST/experiments/model20base2/46260_bundle.pth'
conda activate zerospeech2021
OUTFILE="/worktmp/khorrami/current/ZeroSpeech/output/m20base2E5L1"
mkdir $OUTFILE
zerospeech2021-evaluate /worktmp/khorrami/current/ZeroSpeech/data/  /worktmp/khorrami/current/ZeroSpeech/submission/ -o $OUTFILE  -j12 --no-lexical --no-syntactic --no-semantic --force-cpu
rm -r /worktmp/khorrami/current/ZeroSpeech/submission/phonetic/dev-clean

conda activate fastvgs
python abx_m20.py --mytarget_layer 2 --mytwd '/worktmp/khorrami/current/FaST/experiments/model20base2/46260_bundle.pth'
conda activate zerospeech2021
OUTFILE="/worktmp/khorrami/current/ZeroSpeech/output/m20base2E5L2"
mkdir $OUTFILE
zerospeech2021-evaluate /worktmp/khorrami/current/ZeroSpeech/data/  /worktmp/khorrami/current/ZeroSpeech/submission/ -o $OUTFILE  -j12 --no-lexical --no-syntactic --no-semantic --force-cpu
rm -r /worktmp/khorrami/current/ZeroSpeech/submission/phonetic/dev-clean

conda activate fastvgs
python abx_m20.py --mytarget_layer 3 --mytwd '/worktmp/khorrami/current/FaST/experiments/model20base2/46260_bundle.pth'
conda activate zerospeech2021
OUTFILE="/worktmp/khorrami/current/ZeroSpeech/output/m20base2E5L3"
mkdir $OUTFILE
zerospeech2021-evaluate /worktmp/khorrami/current/ZeroSpeech/data/  /worktmp/khorrami/current/ZeroSpeech/submission/ -o $OUTFILE  -j12 --no-lexical --no-syntactic --no-semantic --force-cpu
rm -r /worktmp/khorrami/current/ZeroSpeech/submission/phonetic/dev-clean

conda activate fastvgs
python abx_m20.py --mytarget_layer 4 --mytwd '/worktmp/khorrami/current/FaST/experiments/model20base2/46260_bundle.pth'
conda activate zerospeech2021
OUTFILE="/worktmp/khorrami/current/ZeroSpeech/output/m20base2E5L4"
mkdir $OUTFILE
zerospeech2021-evaluate /worktmp/khorrami/current/ZeroSpeech/data/  /worktmp/khorrami/current/ZeroSpeech/submission/ -o $OUTFILE  -j12 --no-lexical --no-syntactic --no-semantic --force-cpu
rm -r /worktmp/khorrami/current/ZeroSpeech/submission/phonetic/dev-clean

conda activate fastvgs
python abx_m20.py --mytarget_layer 5 --mytwd '/worktmp/khorrami/current/FaST/experiments/model20base2/46260_bundle.pth'
conda activate zerospeech2021
OUTFILE="/worktmp/khorrami/current/ZeroSpeech/output/m20base2E5L5"
mkdir $OUTFILE
zerospeech2021-evaluate /worktmp/khorrami/current/ZeroSpeech/data/  /worktmp/khorrami/current/ZeroSpeech/submission/ -o $OUTFILE  -j12 --no-lexical --no-syntactic --no-semantic --force-cpu
rm -r /worktmp/khorrami/current/ZeroSpeech/submission/phonetic/dev-clean

conda activate fastvgs
python abx_m20.py --mytarget_layer 1 --mytwd '/worktmp/khorrami/current/FaST/experiments/model20base2/92520_bundle.pth'
conda activate zerospeech2021
OUTFILE="/worktmp/khorrami/current/ZeroSpeech/output/m20base2E10L1"
mkdir $OUTFILE
zerospeech2021-evaluate /worktmp/khorrami/current/ZeroSpeech/data/  /worktmp/khorrami/current/ZeroSpeech/submission/ -o $OUTFILE  -j12 --no-lexical --no-syntactic --no-semantic --force-cpu
rm -r /worktmp/khorrami/current/ZeroSpeech/submission/phonetic/dev-clean

conda activate fastvgs
python abx_m20.py --mytarget_layer 2 --mytwd '/worktmp/khorrami/current/FaST/experiments/model20base2/92520_bundle.pth'
conda activate zerospeech2021
OUTFILE="/worktmp/khorrami/current/ZeroSpeech/output/m20base2E10L2"
mkdir $OUTFILE
zerospeech2021-evaluate /worktmp/khorrami/current/ZeroSpeech/data/  /worktmp/khorrami/current/ZeroSpeech/submission/ -o $OUTFILE  -j12 --no-lexical --no-syntactic --no-semantic --force-cpu
rm -r /worktmp/khorrami/current/ZeroSpeech/submission/phonetic/dev-clean

conda activate fastvgs
python abx_m20.py --mytarget_layer 3 --mytwd '/worktmp/khorrami/current/FaST/experiments/model20base2/92520_bundle.pth'
conda activate zerospeech2021
OUTFILE="/worktmp/khorrami/current/ZeroSpeech/output/m20base2E10L3"
mkdir $OUTFILE
zerospeech2021-evaluate /worktmp/khorrami/current/ZeroSpeech/data/  /worktmp/khorrami/current/ZeroSpeech/submission/ -o $OUTFILE  -j12 --no-lexical --no-syntactic --no-semantic --force-cpu
rm -r /worktmp/khorrami/current/ZeroSpeech/submission/phonetic/dev-clean

conda activate fastvgs
python abx_m20.py --mytarget_layer 4 --mytwd '/worktmp/khorrami/current/FaST/experiments/model20base2/92520_bundle.pth'
conda activate zerospeech2021
OUTFILE="/worktmp/khorrami/current/ZeroSpeech/output/m20base2E10L4"
mkdir $OUTFILE
zerospeech2021-evaluate /worktmp/khorrami/current/ZeroSpeech/data/  /worktmp/khorrami/current/ZeroSpeech/submission/ -o $OUTFILE  -j12 --no-lexical --no-syntactic --no-semantic --force-cpu
rm -r /worktmp/khorrami/current/ZeroSpeech/submission/phonetic/dev-clean

conda activate fastvgs
python abx_m20.py --mytarget_layer 5 --mytwd '/worktmp/khorrami/current/FaST/experiments/model20base2/92520_bundle.pth'
conda activate zerospeech2021
OUTFILE="/worktmp/khorrami/current/ZeroSpeech/output/m20base2E10L5"
mkdir $OUTFILE
zerospeech2021-evaluate /worktmp/khorrami/current/ZeroSpeech/data/  /worktmp/khorrami/current/ZeroSpeech/submission/ -o $OUTFILE  -j12 --no-lexical --no-syntactic --no-semantic --force-cpu
rm -r /worktmp/khorrami/current/ZeroSpeech/submission/phonetic/dev-clean

conda activate fastvgs
python abx_m20.py --mytarget_layer 1 --mytwd '/worktmp/khorrami/current/FaST/experiments/model20base2/138780_bundle.pth'
conda activate zerospeech2021
OUTFILE="/worktmp/khorrami/current/ZeroSpeech/output/m20base2E15L1"
mkdir $OUTFILE
zerospeech2021-evaluate /worktmp/khorrami/current/ZeroSpeech/data/  /worktmp/khorrami/current/ZeroSpeech/submission/ -o $OUTFILE  -j12 --no-lexical --no-syntactic --no-semantic --force-cpu
rm -r /worktmp/khorrami/current/ZeroSpeech/submission/phonetic/dev-clean

conda activate fastvgs
python abx_m20.py --mytarget_layer 2 --mytwd '/worktmp/khorrami/current/FaST/experiments/model20base2/138780_bundle.pth'
conda activate zerospeech2021
OUTFILE="/worktmp/khorrami/current/ZeroSpeech/output/m20base2E15L2"
mkdir $OUTFILE
zerospeech2021-evaluate /worktmp/khorrami/current/ZeroSpeech/data/  /worktmp/khorrami/current/ZeroSpeech/submission/ -o $OUTFILE  -j12 --no-lexical --no-syntactic --no-semantic --force-cpu
rm -r /worktmp/khorrami/current/ZeroSpeech/submission/phonetic/dev-clean

conda activate fastvgs
python abx_m20.py --mytarget_layer 3 --mytwd '/worktmp/khorrami/current/FaST/experiments/model20base2/138780_bundle.pth'
conda activate zerospeech2021
OUTFILE="/worktmp/khorrami/current/ZeroSpeech/output/m20base2E15L3"
mkdir $OUTFILE
zerospeech2021-evaluate /worktmp/khorrami/current/ZeroSpeech/data/  /worktmp/khorrami/current/ZeroSpeech/submission/ -o $OUTFILE  -j12 --no-lexical --no-syntactic --no-semantic --force-cpu
rm -r /worktmp/khorrami/current/ZeroSpeech/submission/phonetic/dev-clean

conda activate fastvgs
python abx_m20.py --mytarget_layer 4 --mytwd '/worktmp/khorrami/current/FaST/experiments/model20base2/138780_bundle.pth'
conda activate zerospeech2021
OUTFILE="/worktmp/khorrami/current/ZeroSpeech/output/m20base2E15L4"
mkdir $OUTFILE
zerospeech2021-evaluate /worktmp/khorrami/current/ZeroSpeech/data/  /worktmp/khorrami/current/ZeroSpeech/submission/ -o $OUTFILE  -j12 --no-lexical --no-syntactic --no-semantic --force-cpu
rm -r /worktmp/khorrami/current/ZeroSpeech/submission/phonetic/dev-clean

conda activate fastvgs
python abx_m20.py --mytarget_layer 5 --mytwd '/worktmp/khorrami/current/FaST/experiments/model20base2/138780_bundle.pth'
conda activate zerospeech2021
OUTFILE="/worktmp/khorrami/current/ZeroSpeech/output/m20base2E15L5"
mkdir $OUTFILE
zerospeech2021-evaluate /worktmp/khorrami/current/ZeroSpeech/data/  /worktmp/khorrami/current/ZeroSpeech/submission/ -o $OUTFILE  -j12 --no-lexical --no-syntactic --no-semantic --force-cpu
rm -r /worktmp/khorrami/current/ZeroSpeech/submission/phonetic/dev-clean


conda activate fastvgs
python abx_m20.py --mytarget_layer 1 --mytwd '/worktmp/khorrami/current/FaST/experiments/model20base2/185040_bundle.pth'
conda activate zerospeech2021
OUTFILE="/worktmp/khorrami/current/ZeroSpeech/output/m20base2E20L1"
mkdir $OUTFILE
zerospeech2021-evaluate /worktmp/khorrami/current/ZeroSpeech/data/  /worktmp/khorrami/current/ZeroSpeech/submission/ -o $OUTFILE  -j12 --no-lexical --no-syntactic --no-semantic --force-cpu
rm -r /worktmp/khorrami/current/ZeroSpeech/submission/phonetic/dev-clean

conda activate fastvgs
python abx_m20.py --mytarget_layer 2 --mytwd '/worktmp/khorrami/current/FaST/experiments/model20base2/185040_bundle.pth'
conda activate zerospeech2021
OUTFILE="/worktmp/khorrami/current/ZeroSpeech/output/m20base2E20L2"
mkdir $OUTFILE
zerospeech2021-evaluate /worktmp/khorrami/current/ZeroSpeech/data/  /worktmp/khorrami/current/ZeroSpeech/submission/ -o $OUTFILE  -j12 --no-lexical --no-syntactic --no-semantic --force-cpu
rm -r /worktmp/khorrami/current/ZeroSpeech/submission/phonetic/dev-clean

conda activate fastvgs
python abx_m20.py --mytarget_layer 3 --mytwd '/worktmp/khorrami/current/FaST/experiments/model20base2/185040_bundle.pth'
conda activate zerospeech2021
OUTFILE="/worktmp/khorrami/current/ZeroSpeech/output/m20base2E20L3"
mkdir $OUTFILE
zerospeech2021-evaluate /worktmp/khorrami/current/ZeroSpeech/data/  /worktmp/khorrami/current/ZeroSpeech/submission/ -o $OUTFILE  -j12 --no-lexical --no-syntactic --no-semantic --force-cpu
rm -r /worktmp/khorrami/current/ZeroSpeech/submission/phonetic/dev-clean

conda activate fastvgs
python abx_m20.py --mytarget_layer 4 --mytwd '/worktmp/khorrami/current/FaST/experiments/model20base2/185040_bundle.pth'
conda activate zerospeech2021
OUTFILE="/worktmp/khorrami/current/ZeroSpeech/output/m20base2E20L4"
mkdir $OUTFILE
zerospeech2021-evaluate /worktmp/khorrami/current/ZeroSpeech/data/  /worktmp/khorrami/current/ZeroSpeech/submission/ -o $OUTFILE  -j12 --no-lexical --no-syntactic --no-semantic --force-cpu
rm -r /worktmp/khorrami/current/ZeroSpeech/submission/phonetic/dev-clean

conda activate fastvgs
python abx_m20.py --mytarget_layer 5 --mytwd '/worktmp/khorrami/current/FaST/experiments/model20base2/185040_bundle.pth'
conda activate zerospeech2021
OUTFILE="/worktmp/khorrami/current/ZeroSpeech/output/m20base2E20L5"
mkdir $OUTFILE
zerospeech2021-evaluate /worktmp/khorrami/current/ZeroSpeech/data/  /worktmp/khorrami/current/ZeroSpeech/submission/ -o $OUTFILE  -j12 --no-lexical --no-syntactic --no-semantic --force-cpu
rm -r /worktmp/khorrami/current/ZeroSpeech/submission/phonetic/dev-clean

