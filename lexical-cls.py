import os
#############################################################################
#twd = '/worktmp/khorrami/current/FaST/experiments/model19base3/9252_bundle.pth'
#target_layer = 2
total_layers = 12
trimTF = True

# Paths for LibriSpeech input and output
wav_path = '/scratch/specog/lextest/data/CDI/'#"/worktmp2/hxkhkh/current/lextest/data/CDI/"
save_path = '/scratch/specog/lextest/embedds/'# "/worktmp2/hxkhkh/current/lextest/embedds/"
os.makedirs(save_path, exist_ok=True)
#############################################################################
# for data
import argparse
import soundfile as sf
import numpy as np
import torch
import json
import numpy
# for model
from models.w2v2_model import  Wav2Vec2Model_cls , ConvFeatureExtractionModel

from steps import trainer
from steps.utils import *
from steps.trainer_utils import *
from models import fast_vgs, w2v2_model
from datasets import spokencoco_dataset, libri_dataset

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

# def LoadAudio( path):
#     audio_feat_len = 8
#     x, sr = sf.read(path, dtype = 'float32')
#     assert sr == 16000
#     length_orig = len(x)
#     if length_orig > 16000 * audio_feat_len:
#         audio_length = int(16000 * audio_feat_len)
#         x = x[:audio_length]
#     else:
#         audio_length = length_orig
#     x_norm = (x - np.mean(x)) / np.std(x)
      
#     return x_norm, audio_length

    
def LoadAudio( path):
    x, sr = sf.read(path, dtype = 'float32')
    assert sr == 16000
    length_orig = len(x)
    audio_length = length_orig
    x_norm = (x - np.mean(x)) / np.std(x)
      
    return x_norm, audio_length

#############################################################################

# writting resamples files

# 
# wav_path = '/worktmp2/hxkhkh/current/lextest/CDI_lextest/CDI_synth/'
# new_path = '/worktmp2/hxkhkh/current/lextest/data/CDI/'
# import librosa
# SAMPLE_RATE = 16000

# for counter, wav_file in enumerate(wav_files_json):
#     y, sr = librosa.load(wav_path + wav_file, dtype = 'float32')
#     x = librosa.resample(y, sr, SAMPLE_RATE)
#     sf.write(new_path + wav_file , x, SAMPLE_RATE)

#############################################################################
# loading Model
device = 'cpu'
# adding all args
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--mytwd", help="my model dir")
parser.add_argument("--mytarget_layer", help="my target layer")
#..............................................................................

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

# input args
mytwd = args.mytwd # "/worktmp2/hxkhkh/current/FaST/exp/E75_bundle.pth"
args.layer_use = int(args.mytarget_layer)


# fixed args
args.encoder_layers = total_layers
args.trim_mask = trimTF
args.normalize = True
args.encoder_attention_heads = 12

print ('###############################')
print(args)
print ('###############################')

############################################## defining the model based on ARGS
#..............................
conv1_trm1_trm3 = Wav2Vec2Model_cls(args)
conv1_trm1_trm3.to(device)
conv1_trm1_trm3.eval()


# loading Pre-trained weights

bundle = torch.load(mytwd)
conv1_trm1_trm3.carefully_load_state_dict(bundle['dual_encoder'])
############################################################################# test

# changing device to gpu
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# conv1_trm1_trm3.to(device)

# conv1_trm1_trm3.eval()
# all_signals = []
# for counter, wav_file in enumerate(wav_files_json):
#     signal_peng,l =  LoadAudio(wav_path + wav_file)
#     all_signals.append(signal_peng)
# audio_signals =  torch.tensor(all_signals ,dtype=torch.float).to(device)  
# input_signals = audio_signals.view(1, -1)
# with torch.no_grad(): 
#     trm13_out = conv1_trm1_trm3(input_signals,  mask=False, features_only=True, tgt_layer=args.layer_use)

    
# trm13_out_features = trm13_out['layer_feats']
# output_tensor = trm13_out_features[0]
# output_np_arr = output_tensor.cpu().detach().numpy()

# for counter, wav_file in enumerate(wav_files_json): 
#     print(counter)   
#     numpy.savetxt(save_path + wav_file [0:-4] + '.txt', output_np_arr[counter] )

#############################################################################

# changing device to gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
conv1_trm1_trm3.to(device)
conv1_trm1_trm3.eval()

wav_files_json = os.listdir(wav_path)

with torch.no_grad():
    for counter, wav_file in enumerate(wav_files_json):
        print(counter)
        signal_peng,l =  LoadAudio(wav_path + wav_file) 
        
        audio_signal = torch.tensor(signal_peng ,dtype=torch.float).to(device)
        input_signal = audio_signal.view(1, -1)
        trm13_out = conv1_trm1_trm3(input_signal,  mask=False, features_only=True, tgt_layer=args.layer_use)
        trm13_out_features = trm13_out['layer_feats']
        output_tensor = trm13_out_features[0] # (time, 768)
        output_np_arr = output_tensor.cpu().detach().numpy()
        #print(len(output_np_arr))
        output_np_arr_cls = numpy.mean(output_np_arr, axis = 0)
        # below line is for saving the results as the first row in text file
        output_np_arr_cls = output_np_arr_cls.reshape(1, output_np_arr_cls.shape[0])
        numpy.savetxt(save_path + wav_file [0:-4] + '.txt', output_np_arr_cls)
        
        torch.cuda.empty_cache()
        #del trm13_out,trm13_out_features,output_tensor,output_np_arr

############################################################################
# from matplotlib import pyplot as plt
# plt.imshow(output_np_arr.T)
# vec = {'embedding_pretrained_model':output_np_arr}
# from scipy.io import savemat
# savemat('/home/hxkhkh/Music/' + "embedding_w2v2_model_layer3.mat", vec)
