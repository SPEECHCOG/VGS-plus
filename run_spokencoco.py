# Author: David Harwath
import argparse
import os
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

# args_0 , unknown_o = parser.parse_args()
# print ('................. we are printing args ..........................')
# logger.info(args_0)
# print ('................. args are printed ..........................')


parser.add_argument("--resume", action="store_true", dest="resume", help="load from exp_dir if True")
parser.add_argument("--validate", action="store_true", default=False, help="temp, if call trainer_variants rather than trainer")
parser.add_argument("--test", action="store_true", default=False, help="test the model on test set")

# args_1 = parser.parse_args()
# print ('................. we are printing args ..........................')
# logger.info(args_1)
# print ('................. args are printed ..........................')

trainer.Trainer.add_args(parser)

w2v2_model.Wav2Vec2Model_cls.add_args(parser)

fast_vgs.DualEncoder.add_args(parser)

spokencoco_dataset.ImageCaptionDataset.add_args(parser)

libri_dataset.LibriDataset.add_args(parser)

args = parser.parse_args()

os.makedirs(args.exp_dir, exist_ok=True)

print('#################### resume #######################')
print(args.resume)

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


if args.validate:
    my_trainer = trainer.Trainer(args)
    my_trainer.validate_one_to_many(hide_progress=False)
else:
    my_trainer = trainer.Trainer(args)
    my_trainer.train()


