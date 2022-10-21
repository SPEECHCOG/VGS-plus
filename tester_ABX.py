# for data
import soundfile as sf
import os
import numpy as np
import torch
import json
import numpy


device = 'cpu'

# for model
from models.w2v2_model import  Wav2Vec2Model_cls , ConvFeatureExtractionModel
import argparse
from steps import trainer
from steps.utils import *
from steps.trainer_utils import *
from models import fast_vgs, w2v2_model
from datasets import spokencoco_dataset, libri_dataset

# Paths for LibriSpeech input and output

wav_path = '/worktmp/khorrami/current/ZeroSpeech/data/phonetic/'
audio_dataset_json_file = '/worktmp/khorrami/current/ZeroSpeech/data/phonetic/index.json'
save_path = '/worktmp/khorrami/current/ZeroSpeech/submission/phonetic/'

# Paths for model weights (traind weights dir)

twd = '/worktmp/khorrami/current/FaST/experiments/model19base3/best_bundle.pth'

#############################################################################

# def LoadAudio( path):
#     audio_feat_len = 8
#     x, sr = sf.read(path, dtype = 'float32')
#     assert sr == 16000
#     length_orig = len(x)
#     if length_orig > 16000 * audio_feat_len:
#         audio_length = int(16000 * audio_feat_len)
#         x = x[:audio_length]
#         x_norm = (x - np.mean(x)) / np.std(x)
#         #x = torch.FloatTensor(x_norm) 
#         x = x_norm
#     else:
#         audio_length = length_orig
#         new_x = np.zeros(int(16000 * audio_feat_len)) #khazar: I change torch to np
#         x_norm = (x - np.mean(x)) / np.std(x)
#         #new_x[:audio_length] = torch.FloatTensor(x_norm)
#         new_x[:audio_length] = x_norm
#         x = new_x
#     return x, audio_length

def LoadAudio( path):
    audio_feat_len = 8
    x, sr = sf.read(path, dtype = 'float32')
    assert sr == 16000
    length_orig = len(x)
    if length_orig > 16000 * audio_feat_len:
        audio_length = int(16000 * audio_feat_len)
        x = x[:audio_length]
    else:
        audio_length = length_orig
    x_norm = (x - np.mean(x)) / np.std(x)
      
    return x_norm, audio_length
#############################################################################
#############################################################################
# loading data

with open(audio_dataset_json_file, 'r') as fp:
    data_json = json.load(fp)
    
test_clean = data_json['subsets']['dev_clean']
wav_files_json = test_clean['items']['wav_list']['files_list']

# signals_peng = []
# for wav_file in wav_files_json:
    
#     signal_peng,l =  LoadAudio(wav_path + wav_file) 
#     signals_peng.append(signal_peng)

# audio_signals = torch.tensor(signals_peng ,dtype=torch.float)
# audio_signals = audio_signals.to(device)
#############################################################################
# loading Model

# adding all args
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
#..............................

# defining the model
args.encoder_layers = 6
args.layer_use = 3
args.trim_mask = True

#..............................
conv1_trm1_trm3 = Wav2Vec2Model_cls(args)
conv1_trm1_trm3.to(device)
conv1_trm1_trm3.eval()


# loading Pre-trained weights

bundle = torch.load(twd)
conv1_trm1_trm3.carefully_load_state_dict(bundle['dual_encoder'])

#############################################################################

# changing device to gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
conv1_trm1_trm3.to(device)
conv1_trm1_trm3.eval()

with torch.no_grad():
    for counter, wav_file in enumerate(wav_files_json):
        print(counter)
        signal_peng,l =  LoadAudio(wav_path + wav_file) 
        
        audio_signal = torch.tensor(signal_peng ,dtype=torch.float).to(device)
        input_signal = audio_signal.view(1, -1)
        trm13_out = conv1_trm1_trm3(input_signal,  mask=False, features_only=True, tgt_layer=4)
        trm13_out_features = trm13_out['layer_feats']
        output_tensor = trm13_out_features[0]
        output_np_arr = output_tensor.cpu().detach().numpy()
        #print(len(output_np_arr))
        numpy.savetxt(save_path + wav_file [0:-4] + '.txt', output_np_arr )
        
        torch.cuda.empty_cache()
        del trm13_out,trm13_out_features,output_tensor,output_np_arr


# from matplotlib import pyplot as plt
# plt.imshow(output_np_arr.T)
# vec = {'embedding_pretrained_model':output_np_arr}
# from scipy.io import savemat
# savemat('/home/hxkhkh/Music/' + "embedding_w2v2_model_layer3.mat", vec)