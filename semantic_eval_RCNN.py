

import torch
torch.cuda.empty_cache()
#%%
# Author: David Harwath
import argparse
import os
import numpy as np
import pickle
import time
from steps import trainer
from models import fast_vgs, w2v2_model
from datasets import spokencoco_dataset, libri_dataset
from logging import getLogger
import logging



logger = getLogger(__name__)
# khazar added below ....
logger.setLevel(logging.DEBUG)
logging.basicConfig()
# .......................

logger.info("I am process %s, running on %s: starting (%s)" % (
        os.getpid(), os.uname()[1], time.asctime()))

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--resume", action="store_true", dest="resume", help="load from exp_dir if True")
parser.add_argument("--validate", action="store_true", default=False, help="temp, if call trainer_variants rather than trainer")
parser.add_argument("--test", action="store_true", default=False, help="test the model on test set")
parser.add_argument("--ssl", action="store_true", dest="ssl", help="only ssl training")

trainer.Trainer.add_args(parser)

w2v2_model.Wav2Vec2Model_cls.add_args(parser)

fast_vgs.DualEncoder.add_args(parser)

spokencoco_dataset.ImageCaptionDataset.add_args(parser)

libri_dataset.LibriDataset.add_args(parser)

args = parser.parse_args()

#%% args from script
save_name_S = "S2b_aL_vB"

exp_dir = '/worktmp2/hxkhkh/current/FaST/experiments/vfplus/exp/'



#..............................................................................
data_root = '/worktmp2/hxkhkh/current/FaST/data'
#raw_audio_base_path = ''
fb_w2v2_weights_fn = '/worktmp2/hxkhkh/current/FaST/model/wav2vec_small.pt'
libri_fn_root = '/worktmp2/hxkhkh/current/FaST/datavf/libri_fn_root/'

args.data_root=data_root
#args.raw_audio_base_path =  raw_audio_base_path
args.fb_w2v2_weights_fn=fb_w2v2_weights_fn
args.exp_dir=exp_dir
args.libri_fn_root=libri_fn_root

    
args.batch_size= 4
args.val_batch_size= 8
args.val_cross_batch_size= 4
args.n_epochs= 50
args.n_print_steps= 100
args.n_val_steps= 1000
args.lr= 0.0001
args.warmup_fraction= 0.1
args.normalize= True
args.xtrm_layers= 1
args.trm_layers= 6
args.fine_matching_weight= 0.0
args.coarse_matching_weight= 1.0
args.libri_w2v2_weight= 0.0
args.caption_w2v2_weight= 1.0
args.feature_grad_mult= 1.0
args.trim_mask= True
args.layer_use= 7
    
#%%

os.makedirs(args.exp_dir, exist_ok=True)

if args.resume or args.validate:
    resume = args.resume
    assert(bool(args.exp_dir))
    with open("%s/args.pkl" % args.exp_dir, "rb") as f:
        old_args = pickle.load(f)
    new_args = vars(args)
    old_args = vars(old_args)
    for key in new_args:
        if key not in old_args or old_args[key] != new_args[key]:
            old_args[key] = new_args[key]
    args = argparse.Namespace(**old_args)
    args.resume = resume
else:
    print("\nexp_dir: %s" % args.exp_dir)
    with open("%s/args.pkl" % args.exp_dir, "wb") as f:
        pickle.dump(args, f)
args.places = False
args.flickr8k = False
args.validate = True
args.test = True
#%%
 
device = 'cpu'
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
my_trainer = trainer.Trainer(args)
batch, s = my_trainer.validate_khazar()

audio = batch['audio'].cpu().detach().numpy()
atm = batch['audio_attention_mask'].cpu().detach().numpy()
al = batch['audio_length'].cpu().detach().numpy()
images = batch ['images'].cpu().detach().numpy()
img_id = batch ['img_id']
fn = batch['fn']

s_np = s.cpu().detach().numpy()

save_path = "/worktmp2/hxkhkh/current/semtest/Smatrix"

np.save( os.path.join(save_path, save_name_S) , s_np)

#%%

import torch
torch.cuda.empty_cache()