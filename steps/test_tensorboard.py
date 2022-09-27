

# import pickle
# file = open('/worktmp/khorrami/current/FaST/exp/progress.pkl', 'rb')
# p = pickle.load(file)
# file.close()

import torch

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd

import matplotlib.pyplot as plt
model5 = torch.load('/worktmp2/hxkhkh/current/FaST/experiments/model5/exp/96000_bundle.pth')



event3 = EventAccumulator('/worktmp2/hxkhkh/current/FaST/experiments/model3/exp/events.out.tfevents.1663923398.nag06.tcsc-local.5309.0')
event4 = EventAccumulator('/worktmp2/hxkhkh/current/FaST/experiments/model4/exp/events.out.tfevents.1663929583.nag03.tcsc-local.15677.0')
event5 = EventAccumulator('/worktmp2/hxkhkh/current/FaST/experiments/model5/exp/events.out.tfevents.1664002919.nag12.tcsc-local.193615.0')
event6 = EventAccumulator('/worktmp2/hxkhkh/current/FaST/experiments/model6/exp/events.out.tfevents.1664140341.nag06.tcsc-local.4384.0')
event6new = EventAccumulator('/worktmp2/hxkhkh/current/FaST/experiments/model6/exp/events.out.tfevents.1664274714.nag06.tcsc-local.24897.0' )




################################   model 1   ##################################
#acc = EventAccumulator("/worktmp/khorrami/current/FaST/experiments/model1/exp/events.out.tfevents.1663792498.nag06.tcsc-local.5854.0")

# tags : 
# 'acc_r10'
# 'coarse_matching_loss'
################################   model 2   ##################################
# event5_r10 = event5.Scalars('acc_r10')
# tags = accreload.Tags()
# scalars = tags['scalars']
# print(scalars)
event3.Reload()
df3 = pd.DataFrame(event3.Scalars('acc_r10'))
x3 = df3['step']
y3 = df3['value']

event4.Reload()
df4 = pd.DataFrame(event4.Scalars('acc_r10'))
x4 = df4['step']
y4 = df4['value']

accreload = event5.Reload()
df5 = pd.DataFrame(event5.Scalars('acc_r10'))
x5 = df5['step']
y5 = df5['value']


accreload = event6.Reload()
df6old = pd.DataFrame(event6.Scalars('acc_r10'))
x6old = df6old['step']
y6old = df6old['value']

accreload = event6new.Reload()
df6new = pd.DataFrame(event6new.Scalars('acc_r10'))
x6new = df6new['step']
y6new = df6new['value']

x6 = pd.concat([x6old,x6new])
y6 = pd.concat([y6old,y6new])
plt.figure()
plt.plot(x3,y3, label = 'model3, VGS-pretrained')
plt.plot(x4,y4, label = 'model4, VGS+-pretrained')
plt.plot(x5,y5, label = 'model5, VGS-random')
plt.plot(x6,y6, label = 'model6, VGS+-random')
plt.grid()
plt.legend()
plt.savefig('figures/recall10-models-all.png', format = 'png')


# plt.figure()
# df4.plot(x='step', y = 'value')
# plt.ylabel("recall@10")
# plt.grid()
# plt.savefig('evaluation_plot_model4.png', format = 'png')


