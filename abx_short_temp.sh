#!/bin/sh
NAME="model7ver7"
OUTFOLDER="/worktmp/khorrami/current/ZeroSpeech/output"/$NAME
mkdir $OUTFOLDER
MFOLDER="/worktmp/khorrami/current/FaST/experiments"/$NAME/"exp"


M="E50_bundle.pth"

OUTNAME="E50L0"
OUTFILE=$OUTFOLDER/$OUTNAME
source activate fastvgs
python abx.py --mytarget_layer 0 --mytwd $MFOLDER/$M
conda activate zerospeech2021
mkdir $OUTFILE
zerospeech2021-evaluate /worktmp/khorrami/current/ZeroSpeech/data/  /worktmp/khorrami/current/ZeroSpeech/submission/ -o $OUTFILE -j12 --no-lexical --no-syntactic --no-semantic --force-cpu
rm -r /worktmp/khorrami/current/ZeroSpeech/submission/phonetic/dev-clean

for LAYERNAME in 1 2 3 4 5 6 7 8 9 10 11
do
    OUTNAME="E50L"$LAYERNAME
    OUTFILE=$OUTFOLDER/$OUTNAME
    conda activate fastvgs
    python abx.py --mytarget_layer $LAYERNAME --mytwd $MFOLDER/$M
    conda activate zerospeech2021
    mkdir $OUTFILE
    zerospeech2021-evaluate /worktmp/khorrami/current/ZeroSpeech/data/  /worktmp/khorrami/current/ZeroSpeech/submission/ -o $OUTFILE -j12 --no-lexical --no-syntactic --no-semantic --force-cpu
    rm -r /worktmp/khorrami/current/ZeroSpeech/submission/phonetic/dev-clean
done

