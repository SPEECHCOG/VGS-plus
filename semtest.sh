#!/bin/sh

source activate fastvgs

ROOT="/worktmp/khorrami/current"
MROOT="/worktmp/khorrami/current/FaST/experiments/vfplus/exp100"

MNAME="expS0"
SNAME="Splus0"
AF="COCO"
python semRCNN.py --exp_dir $MROOT/$MNAME --root $ROOT --Sname $SNAME --afiles $AF

MNAME="expS1"
SNAME="Splus1"
AF="COCO"
python semRCNN.py --exp_dir $MROOT/$MNAME --root $ROOT --Sname $SNAME --afiles $AF

MNAME="expS2"
SNAME="Splus2"
AF="COCO"
python semRCNN.py --exp_dir $MROOT/$MNAME --root $ROOT --Sname $SNAME --afiles $AF

MNAME="expS3"
SNAME="Splus3"
AF="COCO"
python semRCNN.py --exp_dir $MROOT/$MNAME --root $ROOT --Sname $SNAME --afiles $AF

