

# import pickle
# file = open('/worktmp/khorrami/current/FaST/exp/progress.pkl', 'rb')
# p = pickle.load(file)
# file.close()

import torch
model = torch.load('/worktmp/khorrami/current/FaST/experiments/model1/exp/bundle.pth')
kh


from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd



################################   model 1   ##################################
#acc = EventAccumulator("/worktmp/khorrami/current/FaST/experiments/model1/exp/events.out.tfevents.1663792498.nag06.tcsc-local.5854.0")

################################   model 2   ##################################

acc = EventAccumulator("/worktmp/khorrami/current/FaST/experiments/model2/exp/events.out.tfevents.1663796939.nag03.tcsc-local.2866.0")

accreload = acc.Reload()

tags = accreload.Tags()
print(tags)
scalars = tags['scalars']
print(scalars)

acc_r10 = acc.Scalars('acc_r10')

pd.DataFrame(acc.Scalars('acc_r10'))