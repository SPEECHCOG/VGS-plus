# import sys
# import os
# import pickle
# # sys.path.append("the path to FaST-VGS-Family") # might not need this depends on your working dir
# model_path = "/worktmp2/hxkhkh/current/CLD/FaST/model/fast-vgs-coco"
# import torch
# from models import fast_vgs, w2v2_model
# # load args
# with open(f"{model_path}/args.pkl", "rb") as f:
#     args = pickle.load(f)
# # load weights
# weights = torch.load(os.path.join(model_path, "best_bundle.pth"))
# kh
# # if want to use the entire model for e.g. speech-image retrieval (need to first follow section 3 below)
# dual_encoder = fast_vgs.DualEncoder(args)
# cross_encoder = fast_vgs.CrossEncoder(args)
# dual_encoder.load_state_dict(weights['dual_encoder'])
# cross_encoder.load_state_dict(weights['cross_encoder'])

# # if only want to use the audio branch for e.g. feature extraction for speech downstream tasks
# # if you are loading fast-vgs features, it will say that weights of layer 8-11 (0-based) are not seed_dir, that's fine, because fast-vgs only has first 8 layers (i.e. layer 0-7) of w2v2 model, last four layers will be randomly initialized layers
# model = w2v2_model.Wav2Vec2Model_cls(args)
# model.carefully_load_state_dict(weights['dual_encoder']) # will filter out weights that don't belong to w2v2

# # Note that the input to the model should be normalized
# # import SoundFile as sf
# # khazar: I changed this line to soundfile
# import soundfile as sf
# import torch
# import numpy as np
# x, sr = sf.read(wav_path, dtype = 'float32')
# assert sr == 16000
# x_norm = (x - np.mean(x)) / np.std(x)
# x = torch.FloatTensor(x_norm).unsqueeze(0)

# # example of using the audio branch for feature extraction (model is a instance of w2v2_model.Wav2Vec2Model_cls), from layer 7 (0-based)
# model_out = model(source=x, padding_mask=None, mask=False, features_only=True, superb=False, tgt_layer=7)
# # model_out is a dictionary contains cls_token, layer_feats, padding_mask



import base64
import numpy as np
import json
# import cv2
import csv
from tqdm import tqdm

import h5py
from pathlib import Path
import os
import sys
import csv
import base64
import time
import argparse
import numpy as np


csv.field_size_limit(sys.maxsize)
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]

def load_obj_tsv(fname, topk=None):
    """Load object features from tsv file.
    :param fname: The path to the tsv file.
    :param topk: Only load features for top K images (lines) in the tsv file.
        Will load all the features if topk is either -1 or None.
    :return: A list of image object features where each feature is a dict.
        See FILENAMES above for the keys in the feature dict.
    """
    data = []
    start_time = time.time()
    print("Start to load Faster-RCNN detected objects from %s" % fname)
    with open(fname) as f:
        reader = csv.DictReader(f, FIELDNAMES, delimiter="\t")
        for i, item in enumerate(reader):
            for key in ['img_h', 'img_w', 'num_boxes']:
                item[key] = int(item[key])
            boxes = item['num_boxes']
            decode_config = [
                ('objects_id', (boxes, ), np.int64),
                ('objects_conf', (boxes, ), np.float32),
                ('attrs_id', (boxes, ), np.int64),
                ('attrs_conf', (boxes, ), np.float32),
                ('boxes', (boxes, 4), np.float32),
                ('features', (boxes, -1), np.float32),
            ]
            # objects_conf is the prediction confidence ([0,1]) for corresponding objects_id
            # they are ordered descendingly according to confidence (so feats are also order this way, cause they have to match)
            # therefore, attrs_id are not ordered by attrs_conf but objects_conf
            for key, shape, dtype in decode_config:
                item[key] = np.frombuffer(base64.b64decode(item[key]), dtype=dtype)
                item[key] = item[key].reshape(shape)
                item[key].setflags(write=False)
            data.append(item)
            if topk is not None and len(data) == topk:
                break
    elapsed_time = time.time() - start_time
    print("Loaded %d images in file %s in %d seconds." % (len(data), fname, elapsed_time))
    return data
train_audio_dataset_json_file="/worktmp2/hxkhkh/current/FaST/data/coco_pyp/SpokenCOCO/SpokenCOCO_train_unrolled.json"
val_audio_dataset_json_file = "/worktmp2/hxkhkh/current/FaST/data/coco_pyp/SpokenCOCO/SpokenCOCO_val_unrolled.json"
train_img_dataset_tsv_file = "/worktmp2/hxkhkh/current/FaST/data/coco_pyp/coco_img_feat/train2014_obj36.tsv"
val_img_dataset_tsv_file ="/worktmp2/hxkhkh/current/FaST/data/coco_pyp/coco_img_feat/val2014_obj36.tsv"

train_img_dataset_hdf5 = "/worktmp2/hxkhkh/current/FaST/data/coco_pyp/coco_img_feat/SpokenCOCO_train_imgfeat.hdf5"
val_img_dataset_hdf5 = "/worktmp2/hxkhkh/current/FaST/data/coco_pyp/coco_img_feat/SpokenCOCO_val_imgfeat.hdf5"

train_imgid2index_file = "/worktmp2/hxkhkh/current/FaST/data/coco_pyp/SpokenCOCO/SpokenCOCO_train_imgid2idex.json"
val_imgid2index_file = "/worktmp2/hxkhkh/current/FaST/data/coco_pyp/SpokenCOCO/SpokenCOCO_val_imgid2idex.json"

#img_data_train = load_obj_tsv(train_img_dataset_tsv_file)
img_data_val = load_obj_tsv(val_img_dataset_tsv_file)