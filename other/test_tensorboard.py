

# import pickle
# file = open('/worktmp/khorrami/current/FaST/exp/progress.pkl', 'rb')
# p = pickle.load(file)
# file.close()

import torch

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
# model5 = torch.load('/worktmp2/hxkhkh/current/FaST/experiments/model5/exp/96000_bundle.pth')



event3 = EventAccumulator('/worktmp2/hxkhkh/current/FaST/experiments/model3/exp/events.out.tfevents.1663923398.nag06.tcsc-local.5309.0')
event4 = EventAccumulator('/worktmp2/hxkhkh/current/FaST/experiments/model4/exp/events.out.tfevents.1663929583.nag03.tcsc-local.15677.0')
event5_0 = EventAccumulator('/worktmp2/hxkhkh/current/FaST/experiments/model5/exp/events.out.tfevents.1664002919.nag12.tcsc-local.193615.0')
event5_1 = EventAccumulator('/worktmp2/hxkhkh/current/FaST/experiments/model5/exp/events.out.tfevents.1664278096.nag12.tcsc-local.184314.0')
event6_0 = EventAccumulator('/worktmp2/hxkhkh/current/FaST/experiments/model6/exp/events.out.tfevents.1664140341.nag06.tcsc-local.4384.0')
event6_1 = EventAccumulator('/worktmp2/hxkhkh/current/FaST/experiments/model6/exp/events.out.tfevents.1664274714.nag06.tcsc-local.24897.0' )
event6_2 = EventAccumulator('/worktmp2/hxkhkh/current/FaST/experiments/model6/exp/events.out.tfevents.1664405058.nag14.tcsc-local.103335.0')
event6_3 = EventAccumulator('/worktmp2/hxkhkh/current/FaST/experiments/model6/exp/events.out.tfevents.1664458197.nag03.tcsc-local.14499.0')
event6_4 = EventAccumulator('/worktmp2/hxkhkh/current/FaST/experiments/model6/exp/events.out.tfevents.1664521091.nag03.tcsc-local.5612.0')

event6_b = EventAccumulator('/worktmp2/hxkhkh/current/FaST/experiments/model6b/events.out.tfevents.1664639635.r15g02.bullx.29406.0')
event7 = EventAccumulator('/worktmp2/hxkhkh/current/FaST/experiments/model7/events.out.tfevents.1664570487.r13g07.bullx.19210.0')

event7b = EventAccumulator('/worktmp2/hxkhkh/current/FaST/experiments/model7b/events.out.tfevents.1664995320.r04g06.bullx.38494.0')
event8 = EventAccumulator('/worktmp2/hxkhkh/current/FaST/experiments/model8/events.out.tfevents.1664571213.r15g02.bullx.80668.0')
event8b = EventAccumulator('/worktmp2/hxkhkh/current/FaST/experiments/model8b/events.out.tfevents.1664995925.r13g01.bullx.38707.0')
event9 = EventAccumulator('/worktmp2/hxkhkh/current/FaST/experiments/model9/events.out.tfevents.1664999334.nag03.tcsc-local.26041.0') 
event10 = EventAccumulator('/worktmp2/hxkhkh/current/FaST/experiments/model10/events.out.tfevents.1664742364.nag12.tcsc-local.167690.0') 
#event10a = EventAccumulator('/worktmp2/hxkhkh/current/FaST/experiments/model10a/events.out.tfevents.1664911572.nag12.tcsc-local.9983.0') 
event10a = EventAccumulator('/worktmp2/hxkhkh/current/FaST/experiments/model10a/events.out.tfevents.1664912566.nag12.tcsc-local.13095.0') 
event10aa = EventAccumulator('/worktmp2/hxkhkh/current/FaST/experiments/model10aa/events.out.tfevents.1665041287.nag19.tcsc-local.14739.0') 
event11 = EventAccumulator('/worktmp2/hxkhkh/current/FaST/experiments/model11/events.out.tfevents.1664832094.nag02.tcsc-local.6126.0') 
event12a = EventAccumulator('/worktmp2/hxkhkh/current/FaST/experiments/model12a/events.out.tfevents.1664916858.nag02.tcsc-local.5330.0') 

c_3 = 'blue'
c_4 = 'darkorange'
c_5 = 'green'
c_6 = 'red'
c_7 = 'royalblue'
c_8 = 'lightsalmon'
c_9 = 'brown'
c_10 = 'grey'
c_11 = 'pink'
c_12 = 'tan'

spath = '/worktmp2/hxkhkh/current/FaST/experiments/plots/'
n_32 = 18505
n_64 = 9252
kh
#acc = EventAccumulator("/worktmp/khorrami/current/FaST/experiments/model1/exp/events.out.tfevents.1663792498.nag06.tcsc-local.5854.0")

# tags : 
# 'acc_r10'
# 'coarse_matching_loss'
# 'caption_w2v2_loss'
# 'weighted_loss'


# event5_r10 = event5.Scalars('acc_r10')
# tags = accreload.Tags()
# scalars = tags['scalars']
# print(scalars)
################################   model 1   ##################################
# event6_1.Reload()
# df_loss_vgs = pd.DataFrame(event6_1.Scalars('coarse_matching_loss'))
# df_loss_w2v2 = pd.DataFrame(event6_1.Scalars('caption_w2v2_loss'))
# df_loss_total = pd.DataFrame(event6_1.Scalars('weighted_loss'))

# loss_vgs = df_loss_vgs[::50]
# loss_w2v2 = df_loss_w2v2[::50]
# loss_total = df_loss_total[::50]

# y_vgs = loss_vgs['value']
# y_w2v2 = loss_w2v2['value']
# y_total = loss_total['value']

# plt.figure()
# plt.plot(y_vgs, label = 'loss vgs')
# plt.plot(y_w2v2, label = 'loss w2v2')
# plt.plot(y_total, label = 'loss total')

# plt.grid()
# plt.legend()
#plt.savefig('figures/loss-model6new.png', format = 'png')

###############################################################################
                            #    Model 3   #
###############################################################################

event3.Reload()

recall_3 = pd.DataFrame(event3.Scalars('acc_r10'))
x_recall_3 = [ i/n_32 for i in recall_3['step'] ]
y_recall_3 = recall_3['value']


vgsloss_3 = pd.DataFrame(event3.Scalars('coarse_matching_loss'))
x_vgsloss_3 = [ i/n_32 for i in vgsloss_3['step'][::100] ]
y_vgsloss_3 = vgsloss_3['value'][::100]


fig = plt.figure()
fig.suptitle(' model 3, VGS with pretrained weights')
plt.subplot(1,2,1)
plt.plot(x_recall_3,y_recall_3, label = 'recall@10')
plt.xlabel('epoch')
plt.grid()
plt.legend()
plt.subplot(1,2,2)
plt.plot(x_vgsloss_3  , y_vgsloss_3 , label = 'loss vgs')
plt.xlabel('epoch')
plt.grid()
plt.legend()

plt.savefig(spath + 'model3.png', format = 'png')
###############################################################################
                            #    Model 4   #
###############################################################################
event4.Reload()

recall_4 = pd.DataFrame(event4.Scalars('acc_r10'))
x_recall_4 = [ i/n_32 for i in recall_4['step']]
y_recall_4 = recall_4['value']


vgsloss_4 = pd.DataFrame(event4.Scalars('coarse_matching_loss'))
x_vgsloss_4 = [ i/n_32 for i in vgsloss_4['step'][::100] ]
y_vgsloss_4 = vgsloss_4['value'][::100]

caploss_4 = pd.DataFrame(event4.Scalars('caption_w2v2_loss'))
x_caploss_4 = [ i/n_32 for i in caploss_4['step'][::100] ]
y_caploss_4 = caploss_4['value'][::100]

fig = plt.figure()
fig.suptitle(' model 4, VGS+ with pretrained weights')
plt.subplot(1,2,1)
plt.plot(x_recall_4,y_recall_4, label = 'recall@10')
plt.xlabel('epoch')
plt.grid()
plt.legend()
plt.subplot(1,2,2)
plt.plot(x_vgsloss_4  , y_vgsloss_4, label = 'loss vgs')
plt.plot(x_caploss_4  , y_caploss_4, 'olivedrab' ,label = 'loss caption')
plt.xlabel('epoch')
plt.grid()
plt.legend()

plt.savefig(spath + 'model4.png', format = 'png')

###############################################################################
                            #    Model 5   #
###############################################################################
event5_0.Reload()
event5_1.Reload()

recall_5_0 = pd.DataFrame(event5_0.Scalars('acc_r10'))
x_recall_5_0 = [ i/n_32 for i in  recall_5_0['step'] ]
y_recall_5_0 = recall_5_0['value']

vgsloss_5_0 = pd.DataFrame(event5_0.Scalars('coarse_matching_loss'))
x_vgsloss_5_0 = [ i/n_32 for i in vgsloss_5_0['step'][::100]]
y_vgsloss_5_0 = vgsloss_5_0['value'][::100]

# .....................

recall_5_1 = pd.DataFrame(event5_1.Scalars('acc_r10'))
x_recall_5_1 = [ i/n_32 for i in recall_5_1['step'] ]
y_recall_5_1 = recall_5_1['value']

vgsloss_5_1 = pd.DataFrame(event5_1.Scalars('coarse_matching_loss'))
x_vgsloss_5_1 = [ i/n_32 for i in  vgsloss_5_1['step'][::100]]
y_vgsloss_5_1 = vgsloss_5_1['value'][::100]


# .....................

x_recall_5_0_polished = x_recall_5_0 [0:39]
y_recall_5_0_polished = y_recall_5_0 [0:39]
x_recall_5_1_polished = [i-9.48 for i in x_recall_5_1 ]

x_recall_5 = np.concatenate([x_recall_5_0_polished,x_recall_5_1_polished], axis=0)
y_recall_5 = pd.concat([y_recall_5_0_polished,y_recall_5_1], axis=0)
# polishing for VGS
x_vgsloss_5_0_polished = x_vgsloss_5_0[0:48]
y_vgsloss_5_0_polished = y_vgsloss_5_0[0:48]
x_vgsloss_5_1_polished = [i-9.48 for i in x_vgsloss_5_1 ]

x_vgsloss_5 = np.concatenate([x_vgsloss_5_0_polished,x_vgsloss_5_1_polished], axis=0)
y_vgsloss_5 = pd.concat([y_vgsloss_5_0_polished,y_vgsloss_5_1], axis=0)


fig = plt.figure()
fig.suptitle(' model 5, VGS with random weights')
plt.subplot(1,2,1)
plt.plot(x_recall_5,y_recall_5, label = 'recall@10')
plt.xlabel('epoch')
plt.grid()
plt.legend()
plt.subplot(1,2,2)
plt.plot(x_vgsloss_5 ,y_vgsloss_5, label = 'loss vgs')
plt.xlabel('epoch')
plt.grid()
plt.legend()

plt.savefig(spath + 'model5.png', format = 'png')
###############################################################################
                            #    Model 6-a   #
###############################################################################
event6_0.Reload()
event6_1.Reload()
event6_2.Reload()
event6_3.Reload()
event6_4.Reload()

recall_6_0 = pd.DataFrame(event6_0.Scalars('acc_r10'))
x_recall_6_0 = [ i/n_32 for i in  recall_6_0['step'] ]
y_recall_6_0 = recall_6_0['value']

vgsloss_6_0 = pd.DataFrame(event6_0.Scalars('coarse_matching_loss'))
x_vgsloss_6_0 = [ i/n_32 for i in vgsloss_6_0['step'][::100]]
y_vgsloss_6_0 = vgsloss_6_0['value'][::100]

caploss_6_0 = pd.DataFrame(event6_0.Scalars('caption_w2v2_loss'))
x_caploss_6_0 = [ i/n_32 for i in caploss_6_0['step'][::100]]
y_caploss_6_0 = caploss_6_0['value'][::100]

# .....................

recall_6_1 = pd.DataFrame(event6_1.Scalars('acc_r10'))
x_recall_6_1 = [ i/n_32 for i in  recall_6_1['step'] ]
y_recall_6_1 = recall_6_1['value']

vgsloss_6_1 = pd.DataFrame(event6_1.Scalars('coarse_matching_loss'))
x_vgsloss_6_1 = [ i/n_32 for i in vgsloss_6_1['step'][::100]]
y_vgsloss_6_1 = vgsloss_6_1['value'][::100]

caploss_6_1 = pd.DataFrame(event6_1.Scalars('caption_w2v2_loss'))
x_caploss_6_1 = [ i/n_32 for i in caploss_6_1['step'][::100]]
y_caploss_6_1 = caploss_6_1['value'][::100]

# .....................

recall_6_2 = pd.DataFrame(event6_2.Scalars('acc_r10'))
x_recall_6_2 = [ i/n_32 for i in  recall_6_2['step'] ]
y_recall_6_2 = recall_6_2['value']

vgsloss_6_2 = pd.DataFrame(event6_2.Scalars('coarse_matching_loss'))
x_vgsloss_6_2 = [ i/n_32 for i in vgsloss_6_2['step'][::100]]
y_vgsloss_6_2 = vgsloss_6_2['value'][::100]

caploss_6_2 = pd.DataFrame(event6_2.Scalars('caption_w2v2_loss'))
x_caploss_6_2 = [ i/n_32 for i in caploss_6_2['step'][::100]]
y_caploss_6_2 = caploss_6_2['value'][::100]
# .....................

recall_6_3 = pd.DataFrame(event6_3.Scalars('acc_r10'))
x_recall_6_3 = [ i/n_32 for i in  recall_6_3['step'] ]
y_recall_6_3 = recall_6_3['value']

vgsloss_6_3 = pd.DataFrame(event6_3.Scalars('coarse_matching_loss'))
x_vgsloss_6_3 = [ i/n_32 for i in vgsloss_6_3['step'][::100]]
y_vgsloss_6_3 = vgsloss_6_3['value'][::100]

caploss_6_3 = pd.DataFrame(event6_3.Scalars('caption_w2v2_loss'))
x_caploss_6_3 = [ i/n_32 for i in caploss_6_3['step'][::100]]
y_caploss_6_3 = caploss_6_3['value'][::100]

# .....................

recall_6_4 = pd.DataFrame(event6_4.Scalars('acc_r10'))
x_recall_6_4 = [ i/n_32 for i in  recall_6_4['step'] ]
y_recall_6_4 = recall_6_4['value']

vgsloss_6_4 = pd.DataFrame(event6_4.Scalars('coarse_matching_loss'))
x_vgsloss_6_4 = [ i/n_32 for i in vgsloss_6_4['step'][::100]]
y_vgsloss_6_4 = vgsloss_6_4['value'][::100]

caploss_6_4 = pd.DataFrame(event6_4.Scalars('caption_w2v2_loss'))
x_caploss_6_4 = [ i/n_32 for i in caploss_6_4['step'][::100]]
y_caploss_6_4 = caploss_6_4['value'][::100]

# recall_6_4 = pd.DataFrame(event6_4.Scalars('acc_r10'))
# x_recall_6_4 = recall_6_4['step']
# y_recall_6_4 = recall_6_4['value']

# vgsloss_6_4 = pd.DataFrame(event6_4.Scalars('coarse_matching_loss'))
# y_vgsloss_6_4 = vgsloss_6_4['value'][::100]

# caploss_6_4 = pd.DataFrame(event6_4.Scalars('caption_w2v2_loss'))
# y_caploss_6_4 = caploss_6_4['value'][::100]

x_recall_6 = np.concatenate([x_recall_6_0,x_recall_6_1,x_recall_6_2,x_recall_6_3,x_recall_6_4], axis=0)
y_recall_6 = pd.concat([y_recall_6_0,y_recall_6_1,y_recall_6_2, y_recall_6_3, y_recall_6_4], axis=0)

x_vgsloss_6 = np.concatenate([x_vgsloss_6_0,x_vgsloss_6_1,x_vgsloss_6_2,x_vgsloss_6_3,x_vgsloss_6_4], axis=0)
y_vgsloss_6 = pd.concat([y_vgsloss_6_0,y_vgsloss_6_1,y_vgsloss_6_2, y_vgsloss_6_3, y_vgsloss_6_4], axis=0)

x_caploss_6 = np.concatenate([x_caploss_6_0,x_caploss_6_1,x_caploss_6_2,x_caploss_6_3,x_caploss_6_4], axis=0)
y_caploss_6 = pd.concat([y_caploss_6_0,y_caploss_6_1,y_caploss_6_2, y_caploss_6_3, y_caploss_6_4], axis=0)

fig = plt.figure()
fig.suptitle(' model 6-messy, VGS+ with random weights, batch size = 32')
plt.subplot(1,2,1)
plt.plot(x_recall_6,y_recall_6, label = 'recall@10')
plt.grid()
plt.legend()
plt.subplot(1,2,2)
plt.plot(x_vgsloss_6, y_vgsloss_6, label = 'loss vgs')
plt.plot(x_caploss_6, y_caploss_6,'olivedrab', label = 'loss caption')
plt.grid()
plt.legend()

plt.savefig(spath + 'model6-messy.png', format = 'png')

# polishing for recall

x_recall_6_0_polished = x_recall_6_0 [0:20]
x_recall_6_1_polished = x_recall_6_1 [0:8]
x_recall_6_2_polished = x_recall_6_2 [0:12]
x_recall_6_3_polished = x_recall_6_3 [0:12]
x_recall_6_4_polished = x_recall_6_4 

y_recall_6_0_polished = y_recall_6_0 [0:20]
y_recall_6_1_polished = y_recall_6_1 [0:8]
y_recall_6_2_polished = y_recall_6_2 [0:12]
y_recall_6_3_polished = y_recall_6_3 [0:12]


x_recall_6 = np.concatenate([x_recall_6_0_polished,x_recall_6_1_polished,x_recall_6_2_polished,x_recall_6_3_polished,x_recall_6_4_polished], axis=0)
y_recall_6 = pd.concat([y_recall_6_0_polished,y_recall_6_1_polished,y_recall_6_2_polished, y_recall_6_3_polished, y_recall_6_4], axis=0)

# polishing for vgs loss

x_vgsloss_6_0_polished = x_vgsloss_6_0 [0:74]
x_vgsloss_6_1_polished = x_vgsloss_6_1 [0:74]
x_vgsloss_6_2_polished = x_vgsloss_6_2 [0:74]
x_vgsloss_6_3_polished = x_vgsloss_6_3 [0:92]
x_vgsloss_6_4_polished = x_vgsloss_6_4 

y_vgsloss_6_0_polished = y_vgsloss_6_0 [0:74]
y_vgsloss_6_1_polished = y_vgsloss_6_1 [0:74]
y_vgsloss_6_2_polished = y_vgsloss_6_2 [0:74]
y_vgsloss_6_3_polished = y_vgsloss_6_3 [0:92]

x_vgsloss_6 = np.concatenate([x_vgsloss_6_0_polished,x_vgsloss_6_1_polished,x_vgsloss_6_2_polished,x_vgsloss_6_3_polished,x_vgsloss_6_4_polished], axis=0)
y_vgsloss_6 = pd.concat([y_vgsloss_6_0_polished,y_vgsloss_6_1_polished,y_vgsloss_6_2_polished, y_vgsloss_6_3_polished, y_vgsloss_6_4], axis=0)

# polishing for caption loss
x_caploss_6_0_polished = x_caploss_6_0 [0:74]
x_caploss_6_1_polished = x_caploss_6_1 [0:74]
x_caploss_6_2_polished = x_caploss_6_2 [0:74]
x_caploss_6_3_polished = x_caploss_6_3 [0:92]
x_caploss_6_4_polished = x_caploss_6_4

y_caploss_6_0_polished = y_caploss_6_0 [0:74]
y_caploss_6_1_polished = y_caploss_6_1 [0:74]
y_caploss_6_2_polished = y_caploss_6_2 [0:74]
y_caploss_6_3_polished = y_caploss_6_3 [0:92]

x_caploss_6 = np.concatenate([x_caploss_6_0_polished,x_caploss_6_1_polished,x_caploss_6_2_polished,x_caploss_6_3_polished,x_caploss_6_4_polished], axis=0)
y_caploss_6 = pd.concat([y_caploss_6_0_polished,y_caploss_6_1_polished,y_caploss_6_2_polished, y_caploss_6_3_polished, y_caploss_6_4], axis=0)

fig = plt.figure()
fig.suptitle(' model 6-a, VGS+ with random weights, batch size = 32')
plt.subplot(1,2,1)
plt.plot(x_recall_6,y_recall_6, label = 'recall@10')
plt.xlabel('epoch')
plt.grid()
plt.legend()
plt.subplot(1,2,2)
plt.plot(x_vgsloss_6, y_vgsloss_6, label = 'loss vgs')
plt.plot(x_caploss_6, y_caploss_6,'olivedrab', label = 'loss caption')
plt.xlabel('epoch')
plt.grid()
plt.legend()

plt.savefig(spath + 'model6a.png', format = 'png')
###############################################################################
                            #    Model 6-b   #
###############################################################################

event6_b.Reload()


recall_6_b = pd.DataFrame(event6_b.Scalars('acc_r10'))
x_recall_6_b = [ i/n_64 for i in  recall_6_b['step'] ]
y_recall_6_b = recall_6_b['value']

vgsloss_6_b = pd.DataFrame(event6_b.Scalars('coarse_matching_loss'))
x_vgsloss_6_b = [ i/n_64 for i in vgsloss_6_b['step'][::100]]
y_vgsloss_6_b = vgsloss_6_b['value'][::100]

caploss_6_b = pd.DataFrame(event6_b.Scalars('caption_w2v2_loss'))
x_caploss_6_b = [ i/n_64 for i in caploss_6_b['step'][::100]]
y_caploss_6_b = caploss_6_b['value'][::100]

fig = plt.figure()
fig.suptitle(' model 6-b, VGS+ with random weights, batch size = 64')
plt.subplot(1,2,1)
plt.plot(x_recall_6_b,y_recall_6_b, label = 'recall@10')
plt.xlabel('epoch')
plt.grid()
plt.legend()
plt.subplot(1,2,2)
plt.plot(x_vgsloss_6_b ,y_vgsloss_6_b, label = 'loss vgs')
plt.plot(x_caploss_6_b, y_caploss_6_b,'olivedrab', label = 'loss caption')
plt.xlabel('epoch')
plt.grid()
plt.legend()

plt.savefig(spath + 'model6b.png', format = 'png')

###############################################################################
                            #    Model 7   #
###############################################################################

event7.Reload()


recall_7 = pd.DataFrame(event7.Scalars('acc_r10'))
x_recall_7 = [ i/n_32 for i in  recall_7['step'] ]
y_recall_7 = recall_7['value']

vgsloss_7 = pd.DataFrame(event7.Scalars('coarse_matching_loss'))
x_vgsloss_7 = [ i/n_32 for i in vgsloss_7['step'][::100]]
y_vgsloss_7 = vgsloss_7['value'][::100]

caploss_7 = pd.DataFrame(event7.Scalars('caption_w2v2_loss'))
x_caploss_7 = [ i/n_32 for i in caploss_7['step'][::100]]
y_caploss_7 = caploss_7['value'][::100]

fig = plt.figure()
fig.suptitle(' model 7')
plt.subplot(1,2,1)
plt.plot(x_recall_7,y_recall_7, label = 'recall@10')
plt.xlabel('epoch')
plt.grid()
plt.legend()
plt.subplot(1,2,2)
plt.plot(x_vgsloss_7 ,y_vgsloss_7, label = 'loss vgs')
plt.plot(x_caploss_7, y_caploss_7, label = 'loss caption')
plt.xlabel('epoch')
plt.grid()
plt.legend()

plt.savefig(spath + 'model7.png', format = 'png')


###############################################################################
                            #    Model 7b   #
###############################################################################

event7b.Reload()


recall_7b = pd.DataFrame(event7b.Scalars('acc_r10'))
x_recall_7b = [ i/n_64 for i in  recall_7b['step'] ]
y_recall_7b = recall_7b['value']

vgsloss_7b = pd.DataFrame(event7b.Scalars('coarse_matching_loss'))
x_vgsloss_7b = [ i/n_64 for i in vgsloss_7b['step'][::100]]
y_vgsloss_7b = vgsloss_7b['value'][::100]

caploss_7b = pd.DataFrame(event7b.Scalars('caption_w2v2_loss'))
x_caploss_7b = [ i/n_64 for i in caploss_7b['step'][::100]]
y_caploss_7b = caploss_7b['value'][::100]

fig = plt.figure()
fig.suptitle(' model 7b')
plt.subplot(1,2,1)
plt.plot(x_recall_7b,y_recall_7b, label = 'recall@10')
plt.xlabel('epoch')
plt.grid()
plt.legend()
plt.subplot(1,2,2)
plt.plot(x_vgsloss_7b ,y_vgsloss_7b, label = 'loss vgs')
plt.plot(x_caploss_7b, y_caploss_7b, label = 'loss caption')
plt.xlabel('epoch')
plt.grid()
plt.legend()

plt.savefig(spath + 'model7b.png', format = 'png')
###############################################################################
                            #    Model 8   #
###############################################################################

event8.Reload()

recall_8 = pd.DataFrame(event8.Scalars('acc_r10'))
x_recall_8 = [ i/n_32 for i in  recall_8['step'] ]
y_recall_8 = recall_8['value']

vgsloss_8 = pd.DataFrame(event8.Scalars('coarse_matching_loss'))
x_vgsloss_8 = [ i/n_32 for i in vgsloss_8['step'][::100]]
y_vgsloss_8 = vgsloss_8['value'][::100]

caploss_8 = pd.DataFrame(event8.Scalars('caption_w2v2_loss'))
x_caploss_8 = [ i/n_32 for i in caploss_8['step'][::100]]
y_caploss_8 = caploss_8['value'][::100]

fig = plt.figure()
fig.suptitle(' model 8')
plt.subplot(1,2,1)
plt.plot(x_recall_8,y_recall_8, label = 'recall@10')
plt.xlabel('epoch')
plt.grid()
plt.legend()
plt.subplot(1,2,2)
plt.plot(x_vgsloss_8 ,y_vgsloss_8, label = 'loss vgs')
plt.plot(x_caploss_8, y_caploss_8, label = 'loss caption')
plt.xlabel('epoch')
plt.grid()
plt.legend()

plt.savefig(spath + 'model8.png', format = 'png')

###############################################################################
                            #    Model 8 b  #
###############################################################################

event8b.Reload()

recall_8b = pd.DataFrame(event8b.Scalars('acc_r10'))
x_recall_8b = [ i/n_64 for i in  recall_8b['step'] ]
y_recall_8b = recall_8b['value']

vgsloss_8b = pd.DataFrame(event8b.Scalars('coarse_matching_loss'))
x_vgsloss_8b = [ i/n_64 for i in vgsloss_8b['step'][::100]]
y_vgsloss_8b = vgsloss_8b['value'][::100]

caploss_8b = pd.DataFrame(event8b.Scalars('caption_w2v2_loss'))
x_caploss_8b = [ i/n_64 for i in caploss_8b['step'][::100]]
y_caploss_8b = caploss_8b['value'][::100]

fig = plt.figure()
fig.suptitle(' model 8b')
plt.subplot(1,2,1)
plt.plot(x_recall_8b,y_recall_8b, label = 'recall@10')
plt.xlabel('epoch')
plt.grid()
plt.legend()
plt.subplot(1,2,2)
plt.plot(x_vgsloss_8b ,y_vgsloss_8b, label = 'loss vgs')
plt.plot(x_caploss_8b, y_caploss_8b, label = 'loss caption')
plt.xlabel('epoch')
plt.grid()
plt.legend()

plt.savefig(spath + 'model8b.png', format = 'png')
###############################################################################
                            #    Model 9   #
###############################################################################

event9.Reload()

recall_9 = pd.DataFrame(event9.Scalars('acc_r10'))
x_recall_9 = [ i/n_32 for i in  recall_9['step'] ]
y_recall_9 = recall_9['value']

vgsloss_9 = pd.DataFrame(event9.Scalars('coarse_matching_loss'))
x_vgsloss_9 = [ i/n_32 for i in vgsloss_9['step'][::100]]
y_vgsloss_9 = vgsloss_9['value'][::100]

caploss_9 = pd.DataFrame(event9.Scalars('caption_w2v2_loss'))
x_caploss_9 = [ i/n_32 for i in caploss_9['step'][::100]]
y_caploss_9 = caploss_9['value'][::100]

fig = plt.figure()
fig.suptitle(' model 9')
plt.subplot(1,2,1)
plt.plot(x_recall_9,y_recall_9, label = 'recall@10')
plt.grid()
plt.legend()
plt.subplot(1,2,2)
plt.plot(x_vgsloss_9 ,y_vgsloss_9, label = 'loss vgs')
plt.plot(x_caploss_9, y_caploss_9, label = 'loss caption')
plt.grid()
plt.legend()

plt.savefig(spath + 'model9.png', format = 'png')

###############################################################################
                            #    Model 10   #
###############################################################################

event10.Reload()

recall_10_0 = pd.DataFrame(event10.Scalars('acc_r10'))
x_recall_10_0 = [ i/n_32 for i in  recall_10_0['step'] ]
y_recall_10_0 = recall_10_0['value']

vgsloss_10_0 = pd.DataFrame(event10.Scalars('coarse_matching_loss'))
x_vgsloss_10_0 = [ i/n_32 for i in vgsloss_10_0['step'][::100]]
y_vgsloss_10_0 = vgsloss_10_0['value'][::100]

caploss_10_0 = pd.DataFrame(event10.Scalars('caption_w2v2_loss'))
x_caploss_10_0 = [ i/n_32 for i in caploss_10_0['step'][::100]]
y_caploss_10_0 = caploss_10_0['value'][::100]

fig = plt.figure()
fig.suptitle(' model 10_0')
plt.subplot(1,2,1)
plt.plot(x_recall_10_0,y_recall_10_0, label = 'recall@10')
plt.xlabel('epoch')
plt.grid()
plt.legend()
plt.subplot(1,2,2)
plt.plot(x_vgsloss_10_0 ,y_vgsloss_10_0, label = 'loss vgs')
plt.plot(x_caploss_10_0, y_caploss_10_0, label = 'loss caption')
plt.xlabel('epoch')
plt.grid()
plt.legend()

plt.savefig(spath + 'model10_0.png', format = 'png')


################################################################ Model 10 a
event10a.Reload()

recall_10a = pd.DataFrame(event10a.Scalars('acc_r10'))
x_recall_10a = [ i/n_32 for i in  recall_10a['step'] ]
y_recall_10a = recall_10a['value']

vgsloss_10a = pd.DataFrame(event10a.Scalars('coarse_matching_loss'))
x_vgsloss_10a = [ i/n_32 for i in vgsloss_10a['step'][::100]]
y_vgsloss_10a = vgsloss_10a['value'][::100]

caploss_10a = pd.DataFrame(event10a.Scalars('caption_w2v2_loss'))
x_caploss_10a = [ i/n_32 for i in caploss_10a['step'][::100]]
y_caploss_10a = caploss_10a['value'][::100]




################ merging 10-base and 10 a########################### 

x_recall_10 = np.concatenate([x_recall_10_0,x_recall_10a], axis=0)
y_recall_10 = pd.concat([y_recall_10_0,y_recall_10a], axis=0)

x_vgsloss_10 = np.concatenate([x_vgsloss_10_0,x_vgsloss_10a], axis=0)
y_vgsloss_10 = pd.concat([y_vgsloss_10_0,y_vgsloss_10a], axis=0)

x_caploss_10 = np.concatenate([x_caploss_10_0,x_caploss_10a], axis=0)
y_caploss_10 = pd.concat([y_caploss_10_0,y_caploss_10a], axis=0)


fig = plt.figure()
fig.suptitle(' model 10a')
plt.subplot(1,2,1)
plt.plot(x_recall_10,y_recall_10, label = 'recall@10')
plt.xlabel('epoch')
plt.grid()
plt.legend()
plt.subplot(1,2,2)
plt.plot(x_vgsloss_10 ,y_vgsloss_10, label = 'loss vgs')
plt.plot(x_caploss_10, y_caploss_10, label = 'loss caption')
plt.xlabel('epoch')
plt.grid()
plt.legend()

plt.savefig(spath + 'model10a.png', format = 'png')

################################################################ Model 10 aa
event10aa.Reload()

recall_10aa = pd.DataFrame(event10aa.Scalars('acc_r10'))
x_recall_10aa = [ i/n_32 for i in  recall_10aa['step'] ]
y_recall_10aa = recall_10aa['value']

vgsloss_10aa = pd.DataFrame(event10aa.Scalars('coarse_matching_loss'))
x_vgsloss_10aa = [ i/n_32 for i in vgsloss_10aa['step'][::100]]
y_vgsloss_10aa = vgsloss_10aa['value'][::100]

caploss_10aa = pd.DataFrame(event10aa.Scalars('caption_w2v2_loss'))
x_caploss_10aa = [ i/n_32 for i in caploss_10aa['step'][::100]]
y_caploss_10aa = caploss_10aa['value'][::100]


################ merging 10-base and 10 a########################### 

fig = plt.figure()
fig.suptitle(' model 10aa')
plt.subplot(1,2,1)
plt.plot(x_recall_10aa,y_recall_10aa, label = 'recall@10')
plt.xlabel('epoch')
plt.grid()
plt.legend()
plt.subplot(1,2,2)
plt.plot(x_vgsloss_10aa ,y_vgsloss_10aa, label = 'loss vgs')
plt.plot(x_caploss_10aa, y_caploss_10aa, label = 'loss caption')
plt.xlabel('epoch')
plt.grid()
plt.legend()

plt.savefig(spath + 'model10aa.png', format = 'png')

###############################################################################
                            #    Model 11   #
###############################################################################

event11.Reload()

recall_11 = pd.DataFrame(event11.Scalars('acc_r10'))
x_recall_11 = [ i/n_32 for i in  recall_11['step'] ]
y_recall_11 = recall_11['value']

vgsloss_11 = pd.DataFrame(event11.Scalars('coarse_matching_loss'))
x_vgsloss_11 = [ i/n_32 for i in vgsloss_11['step'][::100]]
y_vgsloss_11 = vgsloss_11['value'][::100]

caploss_11 = pd.DataFrame(event11.Scalars('caption_w2v2_loss'))
x_caploss_11 = [ i/n_32 for i in caploss_11['step'][::100]]
y_caploss_11= caploss_11['value'][::100]

fig = plt.figure()
fig.suptitle(' model 11')
plt.subplot(1,2,1)
plt.plot(x_recall_11,y_recall_11, label = 'recall@10')
plt.xlabel('epoch')
plt.xlabel('epoch')
plt.grid()
plt.legend()
plt.subplot(1,2,2)
plt.plot(x_vgsloss_11 ,y_vgsloss_11, label = 'loss vgs')
plt.plot(x_caploss_11, y_caploss_11, label = 'loss caption')
plt.xlabel('epoch')
plt.xlabel('epoch')
plt.grid()
plt.legend()

plt.savefig(spath + 'model11.png', format = 'png')


###############################################################################
                            #    Model 12a   #
###############################################################################

event12a.Reload()

recall_12a = pd.DataFrame(event12a.Scalars('acc_r10'))
x_recall_12a = [ i/n_32 for i in  recall_12a['step'] ]
y_recall_12a = recall_12a['value']

vgsloss_12a = pd.DataFrame(event12a.Scalars('coarse_matching_loss'))
x_vgsloss_12a = [ i/n_32 for i in vgsloss_12a['step'][::100]]
y_vgsloss_12a = vgsloss_12a['value'][::100]

caploss_12a = pd.DataFrame(event12a.Scalars('caption_w2v2_loss'))
x_caploss_12a = [ i/n_32 for i in caploss_12a['step'][::100]]
y_caploss_12a= caploss_12a['value'][::100]

fig = plt.figure()
fig.suptitle(' model 12a')
plt.subplot(1,2,1)
plt.plot(x_recall_12a,y_recall_12a, label = 'recall@10')
plt.xlabel('epoch')
plt.xlabel('epoch')
plt.grid()
plt.legend()
plt.subplot(1,2,2)
plt.plot(x_vgsloss_12a ,y_vgsloss_12a, label = 'loss vgs')
plt.plot(x_caploss_12a, y_caploss_12a, label = 'loss caption')
plt.xlabel('epoch')
plt.xlabel('epoch')
plt.grid()
plt.legend()

plt.savefig(spath + 'model12a.png', format = 'png')
###############################################################################
                            #    plotting   #
###############################################################################

# Recall@10 for replication (model 3,4,5,6) 

plt.figure()
fig.suptitle(' recall@10 for replications of the original VGS and VGS+')
plt.plot(x_recall_3,y_recall_3, c_3, label = 'VGS (Pre)')
plt.plot(x_recall_4,y_recall_4, c_4, label = 'VGS+ (Pre)')
plt.plot(x_recall_5,y_recall_5,c_5, label = 'VGS')
plt.plot(x_recall_6,y_recall_6, c_6, label = 'VGS+')
plt.plot(x_recall_6_b,y_recall_6_b, c_6,linestyle='--', label = 'VGS+, bs=64')
plt.grid()
plt.legend()
plt.ylabel("recall@10")
plt.xlabel("epoch")
plt.savefig(spath + 'recall10-models-3456.png', format = 'png')


plt.figure()
fig.suptitle(' vgs-loss for replications of the original VGS and VGS+')
plt.plot(x_vgsloss_3,y_vgsloss_3, c_3, label = 'VGS (Pre)')
plt.plot(x_vgsloss_4,y_vgsloss_4, c_4, label = 'VGS+ (Pre)')
plt.plot(x_vgsloss_5,y_vgsloss_5, c_5, label = 'VGS')
plt.plot(x_vgsloss_6,y_vgsloss_6, c_6, label = 'VGS+')
plt.plot(x_vgsloss_6_b,y_vgsloss_6_b, c_6,linestyle='--', label = 'VGS+, bs=64')
plt.grid()
plt.legend()
plt.ylabel("vgs-loss")
plt.xlabel("epoch")
plt.savefig(spath + 'vgsloss-models-3456.png', format = 'png')

plt.figure()
fig.suptitle(' caption-loss for replications of the original VGS and VGS+')
plt.plot(x_caploss_4,y_caploss_4, c_4, label = 'VGS+ (Pre)')
plt.plot(x_caploss_6,y_caploss_6,c_6, label = 'VGS+')
plt.plot(x_caploss_6_b,y_caploss_6_b, c_6,linestyle='--', label = 'VGS+, bs=64')
plt.grid()
plt.legend()
plt.ylabel("caption-loss")
plt.xlabel("epoch")
plt.savefig(spath + 'caploss-models-3456.png', format = 'png')

# Recall@10 for bs = 32
#############################################################################
plt.figure()
plt.title('batch_size = 32')
plt.plot(x_recall_6,y_recall_6, c_6, label = 'model 6 (VGS+)')
plt.plot(x_recall_7,y_recall_7, c_7, label = 'model 7')
plt.plot(x_recall_8,y_recall_8, c_8, label = 'model 8')
plt.grid()
plt.legend()
plt.ylabel("recall@10")
plt.xlabel("epoch")
plt.savefig(spath + 'recall10-models-678-a.png', format = 'png')

plt.figure()
plt.title('batch_size = 32')
plt.plot(x_vgsloss_6,y_vgsloss_6, c_6, label = 'model 6 (VGS+)')
plt.plot(x_vgsloss_7,y_vgsloss_7, c_7, label = 'model 7')
plt.plot(x_vgsloss_8,y_vgsloss_8, c_8, label = 'model 8')
plt.grid()
plt.legend()
plt.ylabel("VGS loss")
plt.xlabel("epoch")
plt.savefig(spath + 'vgsloss-models-678-a.png', format = 'png')

plt.figure()
plt.title('batch_size = 32')
plt.plot(x_caploss_6,y_caploss_6, c_6, label = 'model 6 (VGS+)')
plt.plot(x_caploss_7,y_caploss_7, c_7, label = 'model 7')
plt.plot(x_caploss_8,y_caploss_8, c_8,label = 'model 8')
plt.grid()
plt.legend()
plt.ylabel("Caption loss")
plt.xlabel("epoch")
plt.savefig(spath + 'caploss-models-678-a.png', format = 'png')
#############################################################################

# Recall@10 for bs = 64
#############################################################################
plt.figure()
plt.title('batch_size = 64')
plt.plot(x_recall_6_b,y_recall_6_b, c_6,linestyle='--', label = 'model 6b (VGS+)')
plt.plot(x_recall_7b,y_recall_7b, c_7,linestyle='--', label = 'model 7b')
plt.plot(x_recall_8b,y_recall_8b, c_8,linestyle='--', label = 'model 8b')
plt.grid()
plt.legend()
plt.ylabel("recall@10")
plt.xlabel("epoch")
plt.savefig(spath + 'recall10-models-678-b.png', format = 'png')

plt.figure()
plt.title( 'batch_size = 64')
plt.plot(x_vgsloss_6_b,y_vgsloss_6_b, c_6,linestyle='--', label = 'model 6b (VGS+)')
plt.plot(x_vgsloss_7b,y_vgsloss_7b, c_7,linestyle='--', label = 'model 7b')
plt.plot(x_vgsloss_8b,y_vgsloss_8b, c_8,linestyle='--', label = 'model 8b')
plt.grid()
plt.legend()
plt.ylabel("VGS loss")
plt.xlabel("epoch")
plt.savefig(spath + 'vgsloss-models-678-b.png', format = 'png')

plt.figure()
plt.title('batch_size = 64')
plt.plot(x_caploss_6_b,y_caploss_6_b, c_6,linestyle='--', label = 'model 6b (VGS+)')
plt.plot(x_caploss_7b,y_caploss_7b, c_7,linestyle='--', label = 'model 7b')
plt.plot(x_caploss_8b,y_caploss_8b, c_8,linestyle='--', label = 'model 8b')
plt.grid()
plt.legend()
plt.ylabel("Caption loss")
plt.xlabel("epoch")
plt.savefig(spath + 'caploss-models-678-b.png', format = 'png')
#############################################################################

# ALL models for bs = 32
#############################################################################
plt.figure()
plt.title('batch_size = 32')
plt.plot(x_recall_6,y_recall_6, c_6, label = 'model 6 (VGS+)')
plt.plot(x_recall_7,y_recall_7, c_7, label = 'model 7')
plt.plot(x_recall_8,y_recall_8, c_8, label = 'model 8')
plt.plot(x_recall_9,y_recall_9, c_9, label = 'model 9')
plt.plot(x_recall_10,y_recall_10, c_10, label = 'model 10')
plt.plot(x_recall_11,y_recall_11, c_11, label = 'model 11')
plt.grid()
plt.legend()
plt.ylabel("recall@10")
plt.xlabel("epoch")
plt.savefig(spath + 'recall10-models-all-a.png', format = 'png')

plt.figure()
plt.title('batch_size = 32')
plt.plot(x_vgsloss_6,y_vgsloss_6, c_6, label = 'model 6 (VGS+)')
plt.plot(x_vgsloss_7,y_vgsloss_7, c_7, label = 'model 7')
plt.plot(x_vgsloss_8,y_vgsloss_8, c_8, label = 'model 8')
plt.plot(x_vgsloss_9,y_vgsloss_9, c_9, label = 'model 9')
plt.plot(x_vgsloss_10,y_vgsloss_10, c_10, label = 'model 10')
plt.plot(x_vgsloss_11,y_vgsloss_11, c_11, label = 'model 11')
plt.grid()
plt.legend()
plt.ylabel("VGS loss")
plt.xlabel("epoch")
plt.savefig(spath + 'vgsloss-models-all-a.png', format = 'png')

plt.figure()
plt.title('batch_size = 32')
plt.plot(x_caploss_6,y_caploss_6, c_6, label = 'model 6 (VGS+)')
plt.plot(x_caploss_7,y_caploss_7, c_7, label = 'model 7')
plt.plot(x_caploss_8,y_caploss_8, c_8,label = 'model 8')
plt.plot(x_caploss_9,y_caploss_9, c_9,label = 'model 9')
plt.plot(x_caploss_10,y_caploss_10, c_10,label = 'model 10')
plt.plot(x_caploss_11,y_caploss_11, c_11,label = 'model 11')
plt.grid()
plt.legend()
plt.ylabel("Caption loss")
plt.xlabel("epoch")
plt.savefig(spath + 'caploss-models-all-a.png', format = 'png')

plt.figure()
plt.plot(x_recall_3,y_recall_3, label = 'model3')
plt.plot(x_recall_4,y_recall_4, label = 'model4')
plt.plot(x_recall_5,y_recall_5, label = 'model5')
plt.plot(x_recall_6,y_recall_6, label = 'model6a')
plt.plot(x_recall_6_b,y_recall_6_b, label = 'model6b')
plt.plot(x_recall_7,y_recall_7, label = 'model7 ')
plt.plot(x_recall_8,y_recall_8, label = 'model8 ')
plt.plot(x_recall_9,y_recall_9, label = 'model9 ')
plt.plot(x_recall_9,y_recall_9, label = 'model9 ')
plt.plot(x_recall_10,y_recall_10, label = 'model10 ')
plt.plot(x_recall_11,y_recall_11, label = 'model11 ')
plt.plot(x_recall_12a,y_recall_12a, label = 'model12a ')
plt.grid()
plt.legend()
plt.ylabel("recall@10")
plt.xlabel("epoch")
plt.savefig(spath + 'recall10-models-all.png', format = 'png')


# VGS loss for all models 

plt.figure()
plt.plot(x_vgsloss_3, y_vgsloss_3, label = 'model3')
plt.plot(x_vgsloss_4, y_vgsloss_4, label = 'model4')
plt.plot(x_vgsloss_5, y_vgsloss_5, label = 'model5')
plt.plot(x_vgsloss_6, y_vgsloss_6, label = 'model6a')
plt.plot(x_vgsloss_6_b, y_vgsloss_6_b, label = 'model6b')
plt.plot(x_vgsloss_7, y_vgsloss_7, label = 'model7 ')
plt.plot(x_vgsloss_8, y_vgsloss_8, label = 'model8 ')
plt.plot(x_vgsloss_9, y_vgsloss_9, label = 'model9 ')
plt.plot(x_vgsloss_10, y_vgsloss_10, label = 'model10 ')
plt.plot(x_vgsloss_11, y_vgsloss_11, label = 'model11 ')
plt.plot(x_vgsloss_12a, y_vgsloss_12a, label = 'model12a ')
plt.grid()
plt.legend()
plt.ylabel("vgs loss")
plt.xlabel("n_steps")
plt.savefig(spath + 'vgsloss-models-all.png', format = 'png')

# caption loss for all models 

plt.figure()
plt.plot(x_caploss_4,y_caploss_4, label = 'model4')
plt.plot(x_caploss_6, y_caploss_6, label = 'model6a')
plt.plot(x_caploss_6_b, y_caploss_6_b, label = 'model6b')
plt.plot(x_caploss_7, y_caploss_7, label = 'model7 ')
plt.plot(x_caploss_8, y_caploss_8, label = 'model8 ')
plt.plot(x_caploss_9, y_caploss_9, label = 'model9 ')
plt.plot(x_caploss_10, y_caploss_10, label = 'model10 ')
plt.plot(x_caploss_11, y_caploss_11, label = 'model11 ')
plt.plot(x_caploss_12a, y_caploss_12a, label = 'model12a ')
plt.grid()
plt.legend()
plt.ylabel("caption loss")
plt.xlabel("n_steps")
plt.savefig(spath + 'caploss-models-all.png', format = 'png')


# plt.figure()
# df4.plot(x='step', y = 'value')
# plt.ylabel("recall@10")
# plt.grid()
# plt.savefig('evaluation_plot_model4.png', format = 'png')

import torch
bundle = '/worktmp2/hxkhkh/current/FaST/experiments/model4/exp/best_bundle.pth'
w = torch.load(bundle)