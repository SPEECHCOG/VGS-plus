
import soundfile as sf
import os
import numpy as np
import torch
import json


def LoadAudio( path):
    audio_feat_len = 10.
    x, sr = sf.read(path, dtype = 'float32')
    assert sr == 16000
    length_orig = len(x)
    if length_orig > 16000 * audio_feat_len:
        audio_length = int(16000 * audio_feat_len)
        x = x[:audio_length]
        x_norm = (x - np.mean(x)) / np.std(x)
        #x = torch.FloatTensor(x_norm) 
        x = x_norm
    else:
        audio_length = length_orig
        new_x = np.zeros(int(16000 * audio_feat_len)) #khazar: I change torch to np
        x_norm = (x - np.mean(x)) / np.std(x)
        #new_x[:audio_length] = torch.FloatTensor(x_norm)
        new_x[:audio_length] = x_norm
        x = new_x
    return x, audio_length


audio_dataset_json_file = '/worktmp2/hxkhkh/current/ZeroSpeech/data/abxLS/index.json'
with open(audio_dataset_json_file, 'r') as fp:
    data_json = json.load(fp)
    
test_clean = data_json['subsets']['test_clean']
wav_files_json = test_clean['items']['wav_list']['files_list']

wav_path = '/worktmp2/hxkhkh/current/ZeroSpeech/data/abxLS/'
#wav_files = os.listdir(wav_path)
signals = []
signals_peng = []
for wav_file in wav_files_json:
    # print(wav_file)
    # x, sr = sf.read(wav_path + wav_file, dtype = 'float32')
    # signals.append(x)
    
    signal_peng,_ =  LoadAudio(wav_path + wav_file) 
    signals_peng.append(signal_peng)



audio_feats = torch.tensor(signals_peng[0:10])

############################################################################## 
                                # Model #
##############################################################################    


from models.w2v2_model import  Wav2Vec2Model_cls 
import argparse

from steps import trainer
from steps.utils import *
from steps.trainer_utils import *
from models import fast_vgs, w2v2_model
from datasets import spokencoco_dataset, libri_dataset

# ............................adding all args
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--resume", action="store_true", dest="resume", help="load from exp_dir if True")
parser.add_argument("--validate", action="store_true", default=False, help="temp, if call trainer_variants rather than trainer")
parser.add_argument("--test", action="store_true", default=False, help="test the model on test set")
trainer.Trainer.add_args(parser)
w2v2_model.Wav2Vec2Model_cls.add_args(parser)
fast_vgs.DualEncoder.add_args(parser)
spokencoco_dataset.ImageCaptionDataset.add_args(parser)
libri_dataset.LibriDataset.add_args(parser)
args = parser.parse_args()

args.layer_use = 4
#..............................

# defining the model
conv1_trm1_trm3 = Wav2Vec2Model_cls(args)
conv1_trm1_trm3.eval()
print_model_info(conv1_trm1_trm3 , print_model = True)
# loading pre-trained weight

args.fb_w2v2_weights_fn = '/worktmp2/hxkhkh/current/FaST/model/wav2vec_small.pt'
bundle = torch.load(args.fb_w2v2_weights_fn)['model']
conv1_trm1_trm3.carefully_load_state_dict(bundle)

#....................................


trm13_out = conv1_trm1_trm3(audio_feats,  mask=False, features_only=True, tgt_layer=args.layer_use)

output = trm13_out['layer_feats']
