#!/bin/sh
NAME="model7base3T"
OUTFOLDER="/scratch/specog/lextest/output/frame"/$NAME
mkdir $OUTFOLDER
MFOLDER="/scratch/specog/FaST/experiments"/$NAME/"exp"

source activate fastvgs
module load matlab
M="E75_bundle.pth"
for LAYERNAME in 1 2 3 4 5 6 7 8 9 10 11
do
    OUTNAME="E75L"$LAYERNAME
    OUTFILE=$OUTFOLDER/$OUTNAME
    python /scratch/specog/FaST/VGS-plus/lexical.py --mytarget_layer $LAYERNAME --mytwd $MFOLDER/$M
    mkdir $OUTFILE
    cd /scratch/specog/lextest/CDI_lextest/
    sh CDI_lextest.sh '/scratch/specog/lextest/data/CDI/' '/scratch/specog/lextest/embedds/' 'full' 1 $OUTFILE
    rm -r '/scratch/specog/lextest/embedds/'
done

