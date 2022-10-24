#!/bin/sh
source activate fastvgs

python abx_m19.py --mytarget_layer 1 --mytwd '/worktmp/khorrami/current/FaST/experiments/model19base4/9252_bundle.pth'
conda activate zerospeech2021
mkdir /worktmp/khorrami/current/ZeroSpeech/output/m19base4E1L1
zerospeech2021-evaluate /worktmp/khorrami/current/ZeroSpeech/data/  /worktmp/khorrami/current/ZeroSpeech/submission/ -o /worktmp/khorrami/current/ZeroSpeech/output/m19base4E1L1  -j12 --no-lexical --no-syntactic --no-semantic --force-cpu
rm -r /worktmp/khorrami/current/ZeroSpeech/submission/phonetic/dev-clean

conda activate fastvgs
python abx_m19.py --mytarget_layer 2 --mytwd '/worktmp/khorrami/current/FaST/experiments/model19base4/9252_bundle.pth'
conda activate zerospeech2021
mkdir /worktmp/khorrami/current/ZeroSpeech/output/m19base4E1L2
zerospeech2021-evaluate /worktmp/khorrami/current/ZeroSpeech/data/  /worktmp/khorrami/current/ZeroSpeech/submission/ -o /worktmp/khorrami/current/ZeroSpeech/output/m19base4E1L2  -j12 --no-lexical --no-syntactic --no-semantic --force-cpu
rm -r /worktmp/khorrami/current/ZeroSpeech/submission/phonetic/dev-clean

conda activate fastvgs
python abx_m19.py --mytarget_layer 3 --mytwd '/worktmp/khorrami/current/FaST/experiments/model19base4/9252_bundle.pth'
conda activate zerospeech2021
mkdir /worktmp/khorrami/current/ZeroSpeech/output/m19base4E1L3
zerospeech2021-evaluate /worktmp/khorrami/current/ZeroSpeech/data/  /worktmp/khorrami/current/ZeroSpeech/submission/ -o /worktmp/khorrami/current/ZeroSpeech/output/m19base4E1L3  -j12 --no-lexical --no-syntactic --no-semantic --force-cpu
rm -r /worktmp/khorrami/current/ZeroSpeech/submission/phonetic/dev-clean

conda activate fastvgs
python abx_m19.py --mytarget_layer 4 --mytwd '/worktmp/khorrami/current/FaST/experiments/model19base4/9252_bundle.pth'
conda activate zerospeech2021
mkdir /worktmp/khorrami/current/ZeroSpeech/output/m19base4E1L4
zerospeech2021-evaluate /worktmp/khorrami/current/ZeroSpeech/data/  /worktmp/khorrami/current/ZeroSpeech/submission/ -o /worktmp/khorrami/current/ZeroSpeech/output/m19base4E1L4  -j12 --no-lexical --no-syntactic --no-semantic --force-cpu
rm -r /worktmp/khorrami/current/ZeroSpeech/submission/phonetic/dev-clean

conda activate fastvgs
python abx_m19.py --mytarget_layer 5 --mytwd '/worktmp/khorrami/current/FaST/experiments/model19base4/9252_bundle.pth'
conda activate zerospeech2021
mkdir /worktmp/khorrami/current/ZeroSpeech/output/m19base4E1L5
zerospeech2021-evaluate /worktmp/khorrami/current/ZeroSpeech/data/  /worktmp/khorrami/current/ZeroSpeech/submission/ -o /worktmp/khorrami/current/ZeroSpeech/output/m19base4E1L5  -j12 --no-lexical --no-syntactic --no-semantic --force-cpu
rm -r /worktmp/khorrami/current/ZeroSpeech/submission/phonetic/dev-clean


conda activate fastvgs
python abx_m19.py --mytarget_layer 1 --mytwd '/worktmp/khorrami/current/FaST/experiments/model19base4/46260_bundle.pth'
conda activate zerospeech2021
mkdir /worktmp/khorrami/current/ZeroSpeech/output/m19base4E5L1
zerospeech2021-evaluate /worktmp/khorrami/current/ZeroSpeech/data/  /worktmp/khorrami/current/ZeroSpeech/submission/ -o /worktmp/khorrami/current/ZeroSpeech/output/m19base4E5L1  -j12 --no-lexical --no-syntactic --no-semantic --force-cpu
rm -r /worktmp/khorrami/current/ZeroSpeech/submission/phonetic/dev-clean

conda activate fastvgs
python abx_m19.py --mytarget_layer 2 --mytwd '/worktmp/khorrami/current/FaST/experiments/model19base4/46260_bundle.pth'
conda activate zerospeech2021
mkdir /worktmp/khorrami/current/ZeroSpeech/output/m19base4E5L2
zerospeech2021-evaluate /worktmp/khorrami/current/ZeroSpeech/data/  /worktmp/khorrami/current/ZeroSpeech/submission/ -o /worktmp/khorrami/current/ZeroSpeech/output/m19base4E5L2  -j12 --no-lexical --no-syntactic --no-semantic --force-cpu
rm -r /worktmp/khorrami/current/ZeroSpeech/submission/phonetic/dev-clean

conda activate fastvgs
python abx_m19.py --mytarget_layer 3 --mytwd '/worktmp/khorrami/current/FaST/experiments/model19base4/46260_bundle.pth'
conda activate zerospeech2021
mkdir /worktmp/khorrami/current/ZeroSpeech/output/m19base4E5L3
zerospeech2021-evaluate /worktmp/khorrami/current/ZeroSpeech/data/  /worktmp/khorrami/current/ZeroSpeech/submission/ -o /worktmp/khorrami/current/ZeroSpeech/output/m19base4E5L3  -j12 --no-lexical --no-syntactic --no-semantic --force-cpu
rm -r /worktmp/khorrami/current/ZeroSpeech/submission/phonetic/dev-clean

conda activate fastvgs
python abx_m19.py --mytarget_layer 4 --mytwd '/worktmp/khorrami/current/FaST/experiments/model19base4/46260_bundle.pth'
conda activate zerospeech2021
mkdir /worktmp/khorrami/current/ZeroSpeech/output/m19base4E5L4
zerospeech2021-evaluate /worktmp/khorrami/current/ZeroSpeech/data/  /worktmp/khorrami/current/ZeroSpeech/submission/ -o /worktmp/khorrami/current/ZeroSpeech/output/m19base4E5L4  -j12 --no-lexical --no-syntactic --no-semantic --force-cpu
rm -r /worktmp/khorrami/current/ZeroSpeech/submission/phonetic/dev-clean

conda activate fastvgs
python abx_m19.py --mytarget_layer 5 --mytwd '/worktmp/khorrami/current/FaST/experiments/model19base4/46260_bundle.pth'
conda activate zerospeech2021
mkdir /worktmp/khorrami/current/ZeroSpeech/output/m19base4E5L5
zerospeech2021-evaluate /worktmp/khorrami/current/ZeroSpeech/data/  /worktmp/khorrami/current/ZeroSpeech/submission/ -o /worktmp/khorrami/current/ZeroSpeech/output/m19base4E5L5  -j12 --no-lexical --no-syntactic --no-semantic --force-cpu
rm -r /worktmp/khorrami/current/ZeroSpeech/submission/phonetic/dev-clean


conda activate fastvgs
python abx_m19.py --mytarget_layer 1 --mytwd '/worktmp/khorrami/current/FaST/experiments/model19base4/92520_bundle.pth'
conda activate zerospeech2021
mkdir /worktmp/khorrami/current/ZeroSpeech/output/m19base4E10L1
zerospeech2021-evaluate /worktmp/khorrami/current/ZeroSpeech/data/  /worktmp/khorrami/current/ZeroSpeech/submission/ -o /worktmp/khorrami/current/ZeroSpeech/output/m19base4E10L1  -j12 --no-lexical --no-syntactic --no-semantic --force-cpu
rm -r /worktmp/khorrami/current/ZeroSpeech/submission/phonetic/dev-clean

conda activate fastvgs
python abx_m19.py --mytarget_layer 2 --mytwd '/worktmp/khorrami/current/FaST/experiments/model19base4/92520_bundle.pth'
conda activate zerospeech2021
mkdir /worktmp/khorrami/current/ZeroSpeech/output/m19base4E10L2
zerospeech2021-evaluate /worktmp/khorrami/current/ZeroSpeech/data/  /worktmp/khorrami/current/ZeroSpeech/submission/ -o /worktmp/khorrami/current/ZeroSpeech/output/m19base4E10L2  -j12 --no-lexical --no-syntactic --no-semantic --force-cpu
rm -r /worktmp/khorrami/current/ZeroSpeech/submission/phonetic/dev-clean

conda activate fastvgs
python abx_m19.py --mytarget_layer 3 --mytwd '/worktmp/khorrami/current/FaST/experiments/model19base4/92520_bundle.pth'
conda activate zerospeech2021
mkdir /worktmp/khorrami/current/ZeroSpeech/output/m19base4E10L3
zerospeech2021-evaluate /worktmp/khorrami/current/ZeroSpeech/data/  /worktmp/khorrami/current/ZeroSpeech/submission/ -o /worktmp/khorrami/current/ZeroSpeech/output/m19base4E10L3  -j12 --no-lexical --no-syntactic --no-semantic --force-cpu
rm -r /worktmp/khorrami/current/ZeroSpeech/submission/phonetic/dev-clean

conda activate fastvgs
python abx_m19.py --mytarget_layer 4 --mytwd '/worktmp/khorrami/current/FaST/experiments/model19base4/92520_bundle.pth'
conda activate zerospeech2021
mkdir /worktmp/khorrami/current/ZeroSpeech/output/m19base4E10L4
zerospeech2021-evaluate /worktmp/khorrami/current/ZeroSpeech/data/  /worktmp/khorrami/current/ZeroSpeech/submission/ -o /worktmp/khorrami/current/ZeroSpeech/output/m19base4E10L4  -j12 --no-lexical --no-syntactic --no-semantic --force-cpu
rm -r /worktmp/khorrami/current/ZeroSpeech/submission/phonetic/dev-clean

conda activate fastvgs
python abx_m19.py --mytarget_layer 5 --mytwd '/worktmp/khorrami/current/FaST/experiments/model19base4/92520_bundle.pth'
conda activate zerospeech2021
mkdir /worktmp/khorrami/current/ZeroSpeech/output/m19base4E10L5
zerospeech2021-evaluate /worktmp/khorrami/current/ZeroSpeech/data/  /worktmp/khorrami/current/ZeroSpeech/submission/ -o /worktmp/khorrami/current/ZeroSpeech/output/m19base4E10L5  -j12 --no-lexical --no-syntactic --no-semantic --force-cpu
rm -r /worktmp/khorrami/current/ZeroSpeech/submission/phonetic/dev-clean


conda activate fastvgs
python abx_m19.py --mytarget_layer 1 --mytwd '/worktmp/khorrami/current/FaST/experiments/model19base4/138780_bundle.pth'
conda activate zerospeech2021
mkdir /worktmp/khorrami/current/ZeroSpeech/output/m19base4E15L1
zerospeech2021-evaluate /worktmp/khorrami/current/ZeroSpeech/data/  /worktmp/khorrami/current/ZeroSpeech/submission/ -o /worktmp/khorrami/current/ZeroSpeech/output/m19base4E15L1  -j12 --no-lexical --no-syntactic --no-semantic --force-cpu
rm -r /worktmp/khorrami/current/ZeroSpeech/submission/phonetic/dev-clean

conda activate fastvgs
python abx_m19.py --mytarget_layer 2 --mytwd '/worktmp/khorrami/current/FaST/experiments/model19base4/138780_bundle.pth'
conda activate zerospeech2021
mkdir /worktmp/khorrami/current/ZeroSpeech/output/m19base4E15L2
zerospeech2021-evaluate /worktmp/khorrami/current/ZeroSpeech/data/  /worktmp/khorrami/current/ZeroSpeech/submission/ -o /worktmp/khorrami/current/ZeroSpeech/output/m19base4E15L2  -j12 --no-lexical --no-syntactic --no-semantic --force-cpu
rm -r /worktmp/khorrami/current/ZeroSpeech/submission/phonetic/dev-clean

conda activate fastvgs
python abx_m19.py --mytarget_layer 3 --mytwd '/worktmp/khorrami/current/FaST/experiments/model19base4/138780_bundle.pth'
conda activate zerospeech2021
mkdir /worktmp/khorrami/current/ZeroSpeech/output/m19base4E15L3
zerospeech2021-evaluate /worktmp/khorrami/current/ZeroSpeech/data/  /worktmp/khorrami/current/ZeroSpeech/submission/ -o /worktmp/khorrami/current/ZeroSpeech/output/m19base4E15L3  -j12 --no-lexical --no-syntactic --no-semantic --force-cpu
rm -r /worktmp/khorrami/current/ZeroSpeech/submission/phonetic/dev-clean

conda activate fastvgs
python abx_m19.py --mytarget_layer 4 --mytwd '/worktmp/khorrami/current/FaST/experiments/model19base4/138780_bundle.pth'
conda activate zerospeech2021
mkdir /worktmp/khorrami/current/ZeroSpeech/output/m19base4E15L4
zerospeech2021-evaluate /worktmp/khorrami/current/ZeroSpeech/data/  /worktmp/khorrami/current/ZeroSpeech/submission/ -o /worktmp/khorrami/current/ZeroSpeech/output/m19base4E15L4  -j12 --no-lexical --no-syntactic --no-semantic --force-cpu
rm -r /worktmp/khorrami/current/ZeroSpeech/submission/phonetic/dev-clean

conda activate fastvgs
python abx_m19.py --mytarget_layer 5 --mytwd '/worktmp/khorrami/current/FaST/experiments/model19base4/138780_bundle.pth'
conda activate zerospeech2021
mkdir /worktmp/khorrami/current/ZeroSpeech/output/m19base4E15L5
zerospeech2021-evaluate /worktmp/khorrami/current/ZeroSpeech/data/  /worktmp/khorrami/current/ZeroSpeech/submission/ -o /worktmp/khorrami/current/ZeroSpeech/output/m19base4E15L5  -j12 --no-lexical --no-syntactic --no-semantic --force-cpu
rm -r /worktmp/khorrami/current/ZeroSpeech/submission/phonetic/dev-clean


conda activate fastvgs
python abx_m19.py --mytarget_layer 1 --mytwd '/worktmp/khorrami/current/FaST/experiments/model19base4/185040_bundle.pth'
conda activate zerospeech2021
mkdir /worktmp/khorrami/current/ZeroSpeech/output/m19base4E20L1
zerospeech2021-evaluate /worktmp/khorrami/current/ZeroSpeech/data/  /worktmp/khorrami/current/ZeroSpeech/submission/ -o /worktmp/khorrami/current/ZeroSpeech/output/m19base4E20L1  -j12 --no-lexical --no-syntactic --no-semantic --force-cpu
rm -r /worktmp/khorrami/current/ZeroSpeech/submission/phonetic/dev-clean

conda activate fastvgs
python abx_m19.py --mytarget_layer 2 --mytwd '/worktmp/khorrami/current/FaST/experiments/model19base4/185040_bundle.pth'
conda activate zerospeech2021
mkdir /worktmp/khorrami/current/ZeroSpeech/output/m19base4E20L2
zerospeech2021-evaluate /worktmp/khorrami/current/ZeroSpeech/data/  /worktmp/khorrami/current/ZeroSpeech/submission/ -o /worktmp/khorrami/current/ZeroSpeech/output/m19base4E20L2  -j12 --no-lexical --no-syntactic --no-semantic --force-cpu
rm -r /worktmp/khorrami/current/ZeroSpeech/submission/phonetic/dev-clean

conda activate fastvgs
python abx_m19.py --mytarget_layer 3 --mytwd '/worktmp/khorrami/current/FaST/experiments/model19base4/185040_bundle.pth'
conda activate zerospeech2021
mkdir /worktmp/khorrami/current/ZeroSpeech/output/m19base4E20L3
zerospeech2021-evaluate /worktmp/khorrami/current/ZeroSpeech/data/  /worktmp/khorrami/current/ZeroSpeech/submission/ -o /worktmp/khorrami/current/ZeroSpeech/output/m19base4E20L3  -j12 --no-lexical --no-syntactic --no-semantic --force-cpu
rm -r /worktmp/khorrami/current/ZeroSpeech/submission/phonetic/dev-clean

conda activate fastvgs
python abx_m19.py --mytarget_layer 4 --mytwd '/worktmp/khorrami/current/FaST/experiments/model19base4/185040_bundle.pth'
conda activate zerospeech2021
mkdir /worktmp/khorrami/current/ZeroSpeech/output/m19base4E20L4
zerospeech2021-evaluate /worktmp/khorrami/current/ZeroSpeech/data/  /worktmp/khorrami/current/ZeroSpeech/submission/ -o /worktmp/khorrami/current/ZeroSpeech/output/m19base4E20L4  -j12 --no-lexical --no-syntactic --no-semantic --force-cpu
rm -r /worktmp/khorrami/current/ZeroSpeech/submission/phonetic/dev-clean

conda activate fastvgs
python abx_m19.py --mytarget_layer 5 --mytwd '/worktmp/khorrami/current/FaST/experiments/model19base4/185040_bundle.pth'
conda activate zerospeech2021
mkdir /worktmp/khorrami/current/ZeroSpeech/output/m19base4E20L5
zerospeech2021-evaluate /worktmp/khorrami/current/ZeroSpeech/data/  /worktmp/khorrami/current/ZeroSpeech/submission/ -o /worktmp/khorrami/current/ZeroSpeech/output/m19base4E20L5  -j12 --no-lexical --no-syntactic --no-semantic --force-cpu
rm -r /worktmp/khorrami/current/ZeroSpeech/submission/phonetic/dev-clean

