

import pickle
import time
import numpy as np

import logging
logger = logging.getLogger(__name__)

# khazar added below ....
logger.setLevel(logging.DEBUG)
logging.basicConfig()
# .......................



def print_model_info(model, print_model = False, print_params = True):
    if print_model:
        logger.info(model)
    if print_params:
        all_params = {}
        for name, p in model.named_parameters():
            name = name.split(".")[0]
            if name in all_params:
                all_params[name] += p.numel()
            else:
                all_params[name] = p.numel()
        logger.info("num of parameters of each components:")
        for name in all_params:
            logger.info(f"{name}: {all_params[name]/1000000.:.2f}m")



def save_progress(self):
    self.total_progress.append([self.progress['epoch'], self.progress['num_updates'], self.progress['best_step'], self.progress['best_acc'], self.progress['best_libri_val_loss'], time.time() - self.start_time])
    with open("%s/progress.pkl" % self.args.exp_dir, "wb") as f:
        pickle.dump(self.total_progress, f)

def setup_progress(self):
    """
    Need to customize it
    """
    progress = {}
    progress['best_step'] = 1
    progress['best_acc'] = - np.inf
    progress['best_libri_val_loss'] = np.inf
    progress['num_updates'] = 1
    progress['epoch'] = 1
    total_progress = []
    # if self.args.resume or self.args.validate:
    if self.args.resume:
        progress_pkl = "%s/progress.pkl" % self.args.exp_dir
        with open(progress_pkl, "rb") as f:
            total_progress = pickle.load(f)
            progress['epoch'], progress['num_updates'], progress['best_step'], progress['best_acc'], progress['best_libri_val_loss'], _ = total_progress[-1]
        logger.info("\nResume training from:")
        logger.info("  epoch = %s" % progress['epoch'])
        logger.info("  num_updates = %s" % progress['num_updates'])
        logger.info("  best_step = %s" % progress['best_step'])
        logger.info("  best_acc = %s" % progress['best_acc'])
        logger.info("  best_libri_val_loss = %s" % progress['best_libri_val_loss'])
    return progress, total_progress

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count