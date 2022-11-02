#!/bin/sh
NAME="model6base3T"
OUTFOLDER="/worktmp/khorrami/current/ZeroSpeech/output"/$NAME
mkdir $OUTFOLDER
MFOLDER="/worktmp/khorrami/current/FaST/experiments"/$NAME/"exp"

M="E25_bundle.pth"

OUTNAME="E25L1"
OUTFILE=$OUTFOLDER/$OUTNAME
source activate fastvgs
python abx.py --mytarget_layer 1 --mytwd $MFOLDER/$M
conda activate zerospeech2021
mkdir $OUTFILE
zerospeech2021-evaluate /worktmp/khorrami/current/ZeroSpeech/data/  /worktmp/khorrami/current/ZeroSpeech/submission/ -o $OUTFILE  -j12 --no-lexical --no-syntactic --no-semantic --force-cpu
rm -r /worktmp/khorrami/current/ZeroSpeech/submission/phonetic/dev-clean

for LAYERNAME in 2 3 4 5 6
do
    OUTNAME="E25L"$LAYERNAME
    OUTFILE=$OUTFOLDER/$OUTNAME
    conda activate fastvgs
    python abx.py --mytarget_layer $LAYERNAME --mytwd $MFOLDER/$M
    conda activate zerospeech2021
    mkdir $OUTFILE
    zerospeech2021-evaluate /worktmp/khorrami/current/ZeroSpeech/data/  /worktmp/khorrami/current/ZeroSpeech/submission/ -o $OUTFILE  -j12 --no-lexical --no-syntactic --no-semantic --force-cpu
    rm -r /worktmp/khorrami/current/ZeroSpeech/submission/phonetic/dev-clean
done

M="E35_bundle.pth"

for LAYERNAME in 1 2 3 4 5 6
do
    OUTNAME="E35L"$LAYERNAME
    OUTFILE=$OUTFOLDER/$OUTNAME
    conda activate fastvgs
    python abx.py --mytarget_layer $LAYERNAME --mytwd $MFOLDER/$M
    conda activate zerospeech2021
    mkdir $OUTFILE
    zerospeech2021-evaluate /worktmp/khorrami/current/ZeroSpeech/data/  /worktmp/khorrami/current/ZeroSpeech/submission/ -o $OUTFILE  -j12 --no-lexical --no-syntactic --no-semantic --force-cpu
    rm -r /worktmp/khorrami/current/ZeroSpeech/submission/phonetic/dev-clean
done

M="E45_bundle.pth"

for LAYERNAME in 1 2 3 4 5 6
do
    OUTNAME="E45L"$LAYERNAME
    OUTFILE=$OUTFOLDER/$OUTNAME
    conda activate fastvgs
    python abx.py --mytarget_layer $LAYERNAME --mytwd $MFOLDER/$M
    conda activate zerospeech2021
    mkdir $OUTFILE
    zerospeech2021-evaluate /worktmp/khorrami/current/ZeroSpeech/data/  /worktmp/khorrami/current/ZeroSpeech/submission/ -o $OUTFILE  -j12 --no-lexical --no-syntactic --no-semantic --force-cpu
    rm -r /worktmp/khorrami/current/ZeroSpeech/submission/phonetic/dev-clean
done
