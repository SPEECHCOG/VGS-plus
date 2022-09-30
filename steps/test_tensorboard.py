

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
event5_0 = EventAccumulator('/worktmp2/hxkhkh/current/FaST/experiments/model5/exp/events.out.tfevents.1664002919.nag12.tcsc-local.193615.0')
event5_1 = EventAccumulator('/worktmp2/hxkhkh/current/FaST/experiments/model5/exp/events.out.tfevents.1664278096.nag12.tcsc-local.184314.0')
event6_0 = EventAccumulator('/worktmp2/hxkhkh/current/FaST/experiments/model6/exp/events.out.tfevents.1664140341.nag06.tcsc-local.4384.0')
event6_1 = EventAccumulator('/worktmp2/hxkhkh/current/FaST/experiments/model6/exp/events.out.tfevents.1664274714.nag06.tcsc-local.24897.0' )
event6_2 = EventAccumulator('/worktmp2/hxkhkh/current/FaST/experiments/model6/exp/events.out.tfevents.1664405058.nag14.tcsc-local.103335.0')
event6_3 = EventAccumulator('/worktmp2/hxkhkh/current/FaST/experiments/model6/exp/events.out.tfevents.1664458197.nag03.tcsc-local.14499.0')
event6_4 = EventAccumulator('/worktmp2/hxkhkh/current/FaST/experiments/model6/exp/events.out.tfevents.1664521091.nag03.tcsc-local.5612.0')

################################   model 1   ##################################
#acc = EventAccumulator("/worktmp/khorrami/current/FaST/experiments/model1/exp/events.out.tfevents.1663792498.nag06.tcsc-local.5854.0")

# tags : 
# 'acc_r10'
# 'coarse_matching_loss'
# 'caption_w2v2_loss'
# 'weighted_loss'

event6_1.Reload()
df_loss_vgs = pd.DataFrame(event6_1.Scalars('coarse_matching_loss'))
df_loss_w2v2 = pd.DataFrame(event6_1.Scalars('caption_w2v2_loss'))
df_loss_total = pd.DataFrame(event6_1.Scalars('weighted_loss'))

loss_vgs = df_loss_vgs[::50]
loss_w2v2 = df_loss_w2v2[::50]
loss_total = df_loss_total[::50]

y_vgs = loss_vgs['value']
y_w2v2 = loss_w2v2['value']
y_total = loss_total['value']

plt.figure()
plt.plot(y_vgs, label = 'loss vgs')
plt.plot(y_w2v2, label = 'loss w2v2')
plt.plot(y_total, label = 'loss total')

plt.grid()
plt.legend()
#plt.savefig('figures/loss-model6new.png', format = 'png')

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

accreload = event5_0.Reload()
df5_0 = pd.DataFrame(event5_0.Scalars('acc_r10'))
x5_0 = df5_0['step']
y5_0 = df5_0['value']

accreload = event5_1.Reload()
df5_1 = pd.DataFrame(event5_1.Scalars('acc_r10'))
x5_1 = df5_1['step']
y5_1 = df5_1['value']

jump = 4000 * (81-38)
x5_0_polished = x5_0 [0:39]
x5_1_polished = x5_1 -jump
y5_0_polished = y5_0 [0:39]

x5 = pd.concat([x5_0_polished,x5_1_polished])
y5 = pd.concat([y5_0_polished,y5_1])


accreload = event6_0.Reload()
df6_0 = pd.DataFrame(event6_0.Scalars('acc_r10'))
x6_0 = df6_0['step']
y6_0 = df6_0['value']

accreload = event6_1.Reload()
df6_1 = pd.DataFrame(event6_1.Scalars('acc_r10'))
x6_1 = df6_1['step']
y6_1 = df6_1['value']


accreload = event6_2.Reload()
df6_2 = pd.DataFrame(event6_2.Scalars('acc_r10'))
x6_2 = df6_2['step']
y6_2 = df6_2['value']

accreload = event6_3.Reload()
df6_3 = pd.DataFrame(event6_3.Scalars('acc_r10'))
x6_3 = df6_3['step']
y6_3 = df6_3['value']



accreload = event6_4.Reload()
df6_4 = pd.DataFrame(event6_4.Scalars('acc_r10'))
x6_4 = df6_4['step']
y6_4 = df6_4['value']

########## polishing 


# jump1 = 4000 * (26-19)
# x6_0_polished = x6_0 [0:20]


#x6_1_polished = x6_1 -jump1

# jump2 = 4000 * (16-7)
# x6_1_secondpolished = x6_1_polished [0:8]
# x6_2_polished = x6_2 - (jump1 + jump2)

y6_0_polished = y6_0 [0:20]
y6_1_polished = y6_1 [0:8]
y6_2_polished = y6_2 [0:12]
y6_3_polished = y6_3 [0:12]

#jump3 = 4000 * (16-11)
# x6_2_secondpolished = x6_2_polished [0:12]
# x6_3_polished = x6_3 - (jump1 + jump2 + jump3)

x6 = range(0,4000*63,4000)
y6 = pd.concat([y6_0_polished,y6_1_polished,y6_2_polished, y6_3_polished, y6_4])

# ploting

plt.figure()
plt.plot(x3,y3, label = 'model3, VGS-pretrained')
plt.plot(x4,y4, label = 'model4, VGS+-pretrained')
plt.plot(x5,y5, label = 'model5, VGS-random')
plt.plot(x6,y6, label = 'model6, VGS+-random')
plt.grid()
plt.legend()
plt.ylabel("recall@10")
plt.xlabel("n_steps")
#plt.savefig('figures/recall10-models-all.png', format = 'png')


# plt.figure()
# df4.plot(x='step', y = 'value')
# plt.ylabel("recall@10")
# plt.grid()
# plt.savefig('evaluation_plot_model4.png', format = 'png')


