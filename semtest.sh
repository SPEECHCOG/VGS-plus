#!/bin/sh

source activate fastvgs

ROOT="/worktmp/khorrami/current"
GNAME="vfplus"
MROOT="/worktmp/khorrami/current/FaST/experiments/$GNAME/exp6M"
mkdir $ROOT/"semtest"/"Smatrix"/$GNAME


MNAME="expS0"
SNAME=$GNAME/"S0"
AF="COCO"
python semRCNN.py --exp_dir $MROOT/$MNAME --root $ROOT --Sname $SNAME --afiles $AF

MNAME="expS1"
SNAME=$GNAME/"S1"
AF="COCO"
python semRCNN.py --exp_dir $MROOT/$MNAME --root $ROOT --Sname $SNAME --afiles $AF

MNAME="expS2"
SNAME=$GNAME/"S2"
AF="COCO"
python semRCNN.py --exp_dir $MROOT/$MNAME --root $ROOT --Sname $SNAME --afiles $AF

MNAME="expS3"
SNAME=$GNAME/"S3"
AF="COCO"
python semRCNN.py --exp_dir $MROOT/$MNAME --root $ROOT --Sname $SNAME --afiles $AF

