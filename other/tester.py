
import soundfile as sf
import os
import numpy as np
import torch
import json


def LoadAudio( path):
    audio_feat_len = 8
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


#############################################################################
#test_sc
# data_root = '../data/coco_pyp/'
# audio_dataset_json_file = os.path.join(data_root, "SpokenCOCO/SpokenCOCO_train_unrolled_karpathy.json")
# audio_base_path = '../data/audio_SC/'
# with open(audio_dataset_json_file, 'r') as fp:
#     data_json = json.load(fp)
# data = data_json['data']

# all_lengths = []
# all_files = []
# for index in range(len(data)):
#     datum = data[index]
#     wavpath = os.path.join(audio_base_path, datum['caption']['wav']) 
#     signal_peng,l =  LoadAudio(wavpath)
#     if l <= 21000:
#         all_lengths.append(l)
#         all_files.append(wavpath)
    
#############################################################################

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



audio_feats = torch.tensor(signals_peng[0:10],dtype=torch.float)

############################################################################## 
                                # Model #
##############################################################################    


from models.w2v2_model import  Wav2Vec2Model_cls , ConvFeatureExtractionModel
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
#..............................

# defining the model
args.encoder_layers = 6
args.layer_use = 5
args.trim_mask = True

conv1_trm1_trm3 = Wav2Vec2Model_cls(args)
conv1_trm1_trm3.eval()
#dual_encoder = fast_vgs.DualEncoder(args)
###############################################
# to see layers within the model
# conv_feature_extractor = conv1_trm1_trm3.feature_extractor
# model = conv1_trm1_trm3.feature_extractor.conv_layers[0]
# layer0 = getattr(model,'0')
# model = conv1_trm1_trm3.feature_extractor
# args.conv_feature_layers

# feature_enc_layers = eval(args.conv_feature_layers)
##############################################
# convfe  =  ConvFeatureExtractionModel (feature_enc_layers,dropout=0.0, mode=args.extractor_mode, conv_bias=args.conv_bias,)
# model = convfe
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
# model = model().to(device)

# from torchsummary import summary
# summary(model(), input_size=(1,1, 512))
##############################################

# loading pre-trained weight

# conv1_trm1_trm3.eval()
# print_model_info(conv1_trm1_trm3 , print_model = True)

# args.fb_w2v2_weights_fn = '/worktmp2/hxkhkh/current/FaST/model/wav2vec_small.pt'
# bundle = torch.load(args.fb_w2v2_weights_fn)['model']
# conv1_trm1_trm3.carefully_load_state_dict(bundle)

# test this for layer 5 of model base1 (T), model base1 (F), and model base3
twd = '/worktmp2/hxkhkh/current/FaST/experiments/model19base3/best_bundle.pth'
bundle = torch.load(twd)
conv1_trm1_trm3.carefully_load_state_dict(bundle['dual_encoder'])

# calling the w2v2 model on data
trm13_out = conv1_trm1_trm3(audio_feats,  mask=False, features_only=True, tgt_layer=4)
output_torch_tensor = trm13_out['layer_feats']
np_arr = output_torch_tensor.cpu().detach().numpy()
save_folder = '/worktmp2/hxkhkh/current/ZeroSpeech/submission/testtemp/'
fname = save_folder + 'testtemp.txt'
npr = np_arr[0,:,:]
# npr = numpy.array([[1,2,3],[4,5,6]]) # size(2,3) : 2 rows, 3 columns
import numpy
numpy.savetxt(fname, npr)
kh
# for the output of conv feature extractor
model = conv1_trm1_trm3.feature_extractor
t = audio_feats.view(audio_feats.size(0),1, -1)
model_out = model(t)

#....................................
#--conv_feature_layers , default = "[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] + [(512,2,2)]"
# args.feature_grad_mult = 0.1
# args.conv_feature_layers = '[(512, 10, 5)] + [(512, 3, 4)] * 4 + [(512,2,2)] + [(512,2,2)]'
# args.layer_use = 4
# args.encoder_layers = 6
# args.trim_mask = True
# conv1_trm1_trm3 = Wav2Vec2Model_cls(args)

# model = conv1_trm1_trm3.feature_extractor.conv_layers[0]   
# model =  conv1_trm1_trm3
# model_trainable_parameters = filter(lambda p: p.requires_grad, model.parameters())
# params = sum([np.prod(p.size()) for p in model_trainable_parameters]) #93472512 ,95045376, 4m params
# for p in model.parameters(): 
#     print(p.requires_grad)
    
# params of conv layer ~ 4 m
# params with 6 conv layers
# params with 4 conv layers

