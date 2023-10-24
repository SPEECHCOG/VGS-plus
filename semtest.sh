#!/bin/sh

source activate fastvgs

M="RCNN"
G="exp6M"
ROOT="/worktmp/khorrami/current"
MROOT="/worktmp/khorrami/current/FaST/experiments"/$M/$G
mkdir $ROOT/"semtest/S"/$M/$G

MNAME="expS0"
SNAME=$M/$G/"S0_aL_vO"
AF="COCO"
python semRCNN.py --exp_dir $MROOT/$MNAME --root $ROOT --Sname $SNAME --afiles $AF

MNAME="expS1"
SNAME=$M/$G/"S1_aL_vO"
AF="COCO"
python semRCNN.py --exp_dir $MROOT/$MNAME --root $ROOT --Sname $SNAME --afiles $AF

MNAME="expS2"
SNAME=$M/$G/"S2_aL_vO"
AF="COCO"
python semRCNN.py --exp_dir $MROOT/$MNAME --root $ROOT --Sname $SNAME --afiles $AF

MNAME="expS3"
SNAME=$M/$G/"S3_aL_vO"
AF="COCO"
python semRCNN.py --exp_dir $MROOT/$MNAME --root $ROOT --Sname $SNAME --afiles $AF

M="RCNN"
G="expFB"
ROOT="/worktmp/khorrami/current"
MROOT="/worktmp/khorrami/current/FaST/experiments"/$M/$G
mkdir $ROOT/"semtest/S"/$M/$G

MNAME="expS0"
SNAME=$M/$G/"S0_aL_vO"
AF="COCO"
python semRCNN.py --exp_dir $MROOT/$MNAME --root $ROOT --Sname $SNAME --afiles $AF

MNAME="expS1"
SNAME=$M/$G/"S1_aL_vO"
AF="COCO"
python semRCNN.py --exp_dir $MROOT/$MNAME --root $ROOT --Sname $SNAME --afiles $AF

MNAME="expS2"
SNAME=$M/$G/"S2_aL_vO"
AF="COCO"
python semRCNN.py --exp_dir $MROOT/$MNAME --root $ROOT --Sname $SNAME --afiles $AF

MNAME="expS3"
SNAME=$M/$G/"S3_aL_vO"
AF="COCO"
python semRCNN.py --exp_dir $MROOT/$MNAME --root $ROOT --Sname $SNAME --afiles $AF

M="RCNN"
G="expR"
ROOT="/worktmp/khorrami/current"
MROOT="/worktmp/khorrami/current/FaST/experiments"/$M/$G
mkdir $ROOT/"semtest/S"/$M/$G

MNAME="expS0"
SNAME=$M/$G/"S0_aL_vO"
AF="COCO"
python semRCNN.py --exp_dir $MROOT/$MNAME --root $ROOT --Sname $SNAME --afiles $AF

MNAME="expS1"
SNAME=$M/$G/"S1_aL_vO"
AF="COCO"
python semRCNN.py --exp_dir $MROOT/$MNAME --root $ROOT --Sname $SNAME --afiles $AF

MNAME="expS2"
SNAME=$M/$G/"S2_aL_vO"
AF="COCO"
python semRCNN.py --exp_dir $MROOT/$MNAME --root $ROOT --Sname $SNAME --afiles $AF

MNAME="expS3"
SNAME=$M/$G/"S3_aL_vO"
AF="COCO"
python semRCNN.py --exp_dir $MROOT/$MNAME --root $ROOT --Sname $SNAME --afiles $AF

