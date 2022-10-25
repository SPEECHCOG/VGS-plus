import os

import torch

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np

path_source = '/worktmp2/hxkhkh/current/FaST/experiments/'
path_save = '/worktmp2/hxkhkh/current/FaST/experiments/plots/'
# puhti
path_event_6a = 'model6a/events.out.tfevents.1665150679.r01g05.bullx.404982.0'
# I deleted weights of 6b on puhti by mistake
path_event_6b = 'model6b/events.out.tfevents.1665144281.r13g01.bullx.830503.0'
path_event_6bTrim1 = 'model6bTrim/events.out.tfevents.1665332229.r13g02.bullx.1144814.0'
path_event_6bTrim2 = 'model6bTrim/events.out.tfevents.1666434510.r02g03.bullx.2084431.0'
path_event_6bTrim3 = 'model6bTrim/events.out.tfevents.1666452388.r02g03.bullx.2120801.0'

path_event_10bTrim = 'model10bTrim/events.out.tfevents.1665346196.r03g01.bullx.1369259.0'
path_event_18b = 'model18b/events.out.tfevents.1665249277.r03g02.bullx.465911.0'
path_event_18bTrim = 'model18bTrim/events.out.tfevents.1665232396.nag14.tcsc-local.10609.0'
path_event_19bTrim = 'model19bTrim/events.out.tfevents.1665252989.nag12.tcsc-local.154664.0'

########################
path_event_19bT0 = 'model19bT0/events.out.tfevents.1665844641.nag19.tcsc-local.230832.0'
path_event_19bT3 = 'model19bT3/events.out.tfevents.1665591025.nag08.tcsc-local.25712.0'
path_event_19bT4 = 'model19bT4/events.out.tfevents.1665440508.r14g01.bullx.2583427.0'
path_event_19bT5 = 'model19bT5/events.out.tfevents.1665701070.r13g02.bullx.2734031.0'
path_event_19bT6 = 'model19bT6/events.out.tfevents.1665693739.r18g04.bullx.2469046.0'
path_event_19bT7 = 'model19bT7/events.out.tfevents.1665634174.nag14.tcsc-local.25605.0'
path_event_19bT8 = 'model19bT8/events.out.tfevents.1665692696.nag12.tcsc-local.251956.0'

path_event_19base1F = 'model19base1F/events.out.tfevents.1665913494.nag12.tcsc-local.6464.0'
path_event_19base1 = 'model19base1/events.out.tfevents.1665774972.r02g07.bullx.1098917.0'
path_event_19base2 = 'model19base2/events.out.tfevents.1665775653.r01g04.bullx.2394627.0'
path_event_19base3 = 'model19base3/events.out.tfevents.1665775653.r03g01.bullx.1285560.0'
path_event_19base4 = 'model19base4/events.out.tfevents.1666016478.r03g04.bullx.1774863.0'


path_event_20base1 = 'model20base1/'
path_event_20base2 = 'model20base2/'
path_event_20base3 = 'model20base3/'
path_event_20base4 = 'model20base4/'

c_1 = 'blue'
c_2 = 'grey'
c_3 = 'green'
c_4 = 'red'
c_5 = 'royalblue'
c_18 = 'darkorange'
c_19 = 'brown'
c_11 = 'pink'
c_12 = 'tan'

n_32 = 18505
n_64 = 9252

def find_average_train_load_timec(event):
    train_time = pd.DataFrame(event.Scalars('train_time'))['value']
    data_time = pd.DataFrame(event.Scalars('data_time'))['value']
    return train_time.mean , data_time.mean
def find_single_recall (event, n):
    recall = pd.DataFrame(event.Scalars('acc_r10'))
    x_recall = [ i/n for i in recall['step']]
    y_recall = recall['value']
    return x_recall, y_recall.to_list()

def find_single_lr (event, n , interval):
    lr = pd.DataFrame(event.Scalars('lr'))
    x_lr = [ i/n for i in lr['step'][::interval] ]
    y_lr = lr['value'][::interval]
    return x_lr, y_lr

def find_single_vgsloss (event, n, interval):
    vgsloss = pd.DataFrame(event.Scalars('coarse_matching_loss'))
    x_vgsloss = [ i/n for i in vgsloss['step'][::interval] ]
    y_vgsloss = vgsloss['value'][::interval]
    y_vgsloss_list = y_vgsloss.to_list()
    return x_vgsloss, y_vgsloss_list
 
def find_single_caploss (event, n , interval):
    caploss = pd.DataFrame(event.Scalars('caption_w2v2_loss'))
    x_caploss = [ i/n for i in caploss['step'][::interval] ]
    y_caploss = caploss['value'][::interval]
    y_caploss_list = y_caploss.to_list()
    return x_caploss, y_caploss_list

def plot_single_event(event,n , name , title):

    x_recall, y_recall = find_single_recall (event, n)
    x_lr, y_lr = find_single_lr (event, n)
    x_vgsloss, y_vgsloss = find_single_vgsloss (event, n)
    x_caploss, y_caploss = find_single_caploss (event, n)
    fig = plt.figure()
    fig.suptitle(title)
    plt.subplot(1,2,1)
    plt.plot(x_recall,y_recall, label = 'recall@10')
    y_baseline = 0.793 * np.ones(len(x_recall))
    plt.plot(x_recall,y_baseline, 'gray',label = 'Peng & Harwath')
    #plt.plot(x_lr,y_lr, label = 'lr')
    plt.xlabel('epoch')
    plt.grid()
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(x_vgsloss  , y_vgsloss, label = 'loss vgs')
    plt.plot(x_caploss  , y_caploss, label = 'loss caption')
    plt.xlabel('epoch')
    plt.grid()
    plt.legend()

    plt.savefig(os.path.join(path_save , name + '.png'), format = 'png')
    return x_caploss, y_caploss

def plot_double_events(event1,event2, label1 , label2, c1,c2, n, pltname, title):

    x1_recall, y1_recall = find_single_recall (event1, n)   
    x1_vgsloss, y1_vgsloss = find_single_vgsloss (event1, n, 100)
    x1_caploss, y1_caploss = find_single_caploss (event1, n, 100)
    
    x2_recall, y2_recall = find_single_recall (event2, n)   
    x2_vgsloss, y2_vgsloss = find_single_vgsloss (event2, n, 100)
    x2_caploss, y2_caploss = find_single_caploss (event2, n, 100)
    
    fig = plt.figure(figsize=(15,10))
    fig.suptitle(title, fontsize=22)
    plt.subplot(1,3,1)
    plt.plot(x1_recall,y1_recall, c1, label =  label1 + ' ... ' + str (round (max(y1_recall), 3)) )
    plt.plot(x2_recall,y2_recall, c2, label = label2 + ' ... ' + str (round (max(y2_recall), 3)) )
    #plt.plot(x_lr,y_lr, label = 'lr')
    plt.xlabel('epoch',size=16)
    plt.ylabel('recall@10',size=16)
    plt.ylim(0,0.9,0.1)
    plt.yticks(np.arange(0, 0.9, step=0.1)) 
    plt.grid()
    plt.legend(fontsize=18)
    plt.subplot(1,3,2)
    plt.plot(x1_vgsloss  , y1_vgsloss, c1, label = label1 + ' ... ' + str ( round (min(y1_vgsloss),3)))
    plt.plot(x2_vgsloss  , y2_vgsloss, c2, label = label2 + ' ... ' + str (round (min(y2_vgsloss), 3)) )
    plt.xlabel('epoch',size=16)
    plt.ylabel('loss-vgs',size=16)
    plt.ylim(0,14,1)
    plt.yticks(np.arange(0, 14, step=1)) 
    plt.grid()
    plt.legend(fontsize=18)
    plt.subplot(1,3,3)
    plt.plot(x1_caploss  , y1_caploss, c1, label = label1 + ' ... ' + str (round (min(y1_caploss),3)))
    plt.plot(x2_caploss  , y2_caploss, c2, label = label2 + ' ... ' + str (round (min(y2_caploss), 3)))
    plt.xlabel('epoch',size=16)
    plt.ylabel('loss-caption',size=16)
    plt.ylim(0,14,1)
    plt.yticks(np.arange(0, 14, step=1)) 
    plt.grid()
    plt.legend(fontsize=18)

    plt.savefig(os.path.join(path_save , pltname + '.png'), format = 'png')
    
    
###############################################################################
kh

# event 6bTrim
event_6bTrim1 =  EventAccumulator(os.path.join(path_source, path_event_6bTrim1))
event_6bTrim1.Reload()

event_6bTrim2 =  EventAccumulator(os.path.join(path_source, path_event_6bTrim2))
event_6bTrim2.Reload()

event_6bTrim3 =  EventAccumulator(os.path.join(path_source, path_event_6bTrim3))
event_6bTrim3.Reload()



# plot_single_event(event_6bTrim1 , n_64 , 'model6bTrim' , 'VGS+ (Trim-mask), random init, bs = 64' )  

# event 18b
event_18b =  EventAccumulator(os.path.join(path_source, path_event_18b))
event_18b.Reload()
plot_single_event(event_18b , n_64 , 'model18b' , 'VGS+light, model 18b' ) 

# event 18bTrim
event_18bTrim =  EventAccumulator(os.path.join(path_source, path_event_18bTrim))
event_18bTrim.Reload()
plot_single_event(event_18bTrim , n_64 , 'model18bTrim' , 'VGS+light(Trim-mask), model 18b' ) 

# event 19bTrim
event_19bTrim =  EventAccumulator(os.path.join(path_source, path_event_19bTrim))
event_19bTrim.Reload()
x_recall, y_recall = plot_single_event(event_19bTrim , n_64 , 'model19bTrim' , 'VGS+light(Trim-mask), model 19b' ) 

###########################
######### Base models
event_19base1 =  EventAccumulator(os.path.join(path_source, path_event_19base1))
event_19base1.Reload()
x_recall, y_recall = plot_single_event(event_19base1 , n_64 , 'model19base1' , '19base1... w2v2 (min caption_loss: 1.041)' ) 

event_19base2 =  EventAccumulator(os.path.join(path_source, path_event_19base2))
event_19base2.Reload()
x_recall, y_recall = plot_single_event(event_19base2 , n_64 , 'model19base2' , '19base2... VGS' ) 

event_19base3 =  EventAccumulator(os.path.join(path_source, path_event_19base3))
event_19base3.Reload()
x_recall, y_recall = plot_single_event(event_19base3 , n_64 , 'model19base3' , '19base3... VGS+ (min caption_loss: 1.309)' ) 

event_19base4 =  EventAccumulator(os.path.join(path_source, path_event_19base4))
event_19base4.Reload()
x_recall, y_recall = plot_single_event(event_19base4 , n_64 , 'model19base4' , '19base4... VGS+-pretrained ' ) 


event_19base1F =  EventAccumulator(os.path.join(path_source, path_event_19base1F))
event_19base1F.Reload()
x_recall, y_recall = plot_single_event(event_19base1F , n_64 , 'model19base1F' , '19base1F... w2v2 (min caption_loss: 2.81)' ) 


plot_double_events(event_19base1, event_19base3, 'w2v2 ','VGS+ ','blue','green', n_64, 'm19b1b3','baseline1 (w2v2) versus baseline3 (VGS+)')
plot_double_events(event_19base2, event_19base3, 'VGS ','VGS+ ','blue','green', n_64, 'm19b2b3','baseline2 (VGS) versus baseline3 (VGS+)')

plot_double_events(event_19base1, event_19base1F, 'w2v2 Trim ','w2v2 not Trim ','blue','green', n_64, 'm19b1Tb1F','baseline1T (w2v2) versus baseline1-not Trim (w2v2)')

plot_double_events(event_19base4, event_19base3, 'VGS+ Pre ','VGS+ Sim ','blue','green', n_64, 'm19b3b4','Pretrained versus simultaneous training')


################################################################# Recalls
event_19base2 =  EventAccumulator(os.path.join(path_source, path_event_19base2))
event_19base2.Reload()
x19_base2_recall, y19_base2_recall = find_single_recall(event_19base2, n_64)

event_19base3 =  EventAccumulator(os.path.join(path_source, path_event_19base3))
event_19base3.Reload()
x19_base3_recall, y19_base3_recall = find_single_recall(event_19base3, n_64)

event_19base4 =  EventAccumulator(os.path.join(path_source, path_event_19base4))
event_19base4.Reload()
x19_base4_recall, y19_base4_recall = find_single_recall(event_19base4, n_64)

# ecall for the reference model
x6bTrim1_recall, y6bTrim1_recall = find_single_recall(event_6bTrim1, n_64)
#x6bTrim2_recall, y6bTrim2_recall = find_single_recall(event_6bTrim2, n_64)
x6bTrim3_recall, y6bTrim3_recall = find_single_recall(event_6bTrim3, n_64)

y6bTrim_recall = np.append(y6bTrim1_recall,y6bTrim3_recall[2:])
################################################################# VGS loss
i = 500
event_19base2 =  EventAccumulator(os.path.join(path_source, path_event_19base2))
event_19base2.Reload()
x19_base2_vgsloss, y19_base2_vgsloss = find_single_vgsloss(event_19base2, n_64 , i)

event_19base3 =  EventAccumulator(os.path.join(path_source, path_event_19base3))
event_19base3.Reload()
x19_base3_vgsloss, y19_base3_vgsloss = find_single_vgsloss(event_19base3, n_64, i)

event_19base4 =  EventAccumulator(os.path.join(path_source, path_event_19base4))
event_19base4.Reload()
x19_base4_vgsloss, y19_base4_vgsloss = find_single_vgsloss(event_19base4, n_64, i)

################################################################# caption loss
i = 500
event_19base1 =  EventAccumulator(os.path.join(path_source, path_event_19base1))
event_19base1.Reload()
x19_base1_caploss, y19_base1_caploss = find_single_caploss(event_19base1, n_64 , i)

event_19base3 =  EventAccumulator(os.path.join(path_source, path_event_19base3))
event_19base3.Reload()
x19_base3_caploss, y19_base3_caploss = find_single_caploss(event_19base3, n_64, i)

event_19base4 =  EventAccumulator(os.path.join(path_source, path_event_19base4))
event_19base4.Reload()
x19_base4_caploss, y19_base4_caploss = find_single_caploss(event_19base4, n_64, i)

################################################################# plotting recall and loss for model 19base
title = 'model light, with gradmul = 0.1'
label1 = 'w2v2'
label2 = 'VGS'
label3 = 'VGS+'
label4 = 'VGS+ pretrained'
label5 = 'VGS+ reference'
fig = plt.figure(figsize=(15,10))
fig.suptitle(title, fontsize=20)
# recall
plt.subplot(1,3,1)
plt.plot(x19_base2_recall, y19_base2_recall, c_2, label = label2)
plt.plot(x19_base3_recall, y19_base3_recall, c_3, label = label3)
plt.plot(x19_base4_recall, y19_base4_recall, c_4, label = label4)
plt.plot(x19_base2_recall [0:20] ,y6bTrim_recall, c_5, label = label5)
plt.xlabel('epoch',size=18)
plt.ylabel('recall@10',size=18)
plt.ylim(0,0.8)
plt.grid()
plt.legend(fontsize=16)
#vgs loss
plt.subplot(1,3,2)
plt.plot(x19_base2_vgsloss, y19_base2_vgsloss, c_2, label = label2)
plt.plot(x19_base3_vgsloss, y19_base3_vgsloss, c_3, label = label3)
plt.plot(x19_base4_vgsloss, y19_base4_vgsloss, c_4, label = label4)
plt.xlabel('epoch',size=18)
plt.ylabel('VGS loss',size=18)
plt.ylim(0,14)
plt.grid()
plt.legend(fontsize=16)
# cap loss
plt.subplot(1,3,3)
plt.plot(x19_base1_caploss, y19_base1_caploss, c_1, label = label1)
plt.plot(x19_base3_caploss, y19_base3_caploss, c_3, label = label3)
plt.plot(x19_base4_caploss, y19_base4_caploss, c_4, label = label4)
plt.xlabel('epoch',size=18)
plt.ylabel('lcaption loss',size=18)
plt.ylim(0,14)
plt.grid()
plt.legend(fontsize=16)
plt.savefig(os.path.join(path_save , 'fig1_light19_recall_loss-add' + '.png'), format = 'png')

################################################################# plotting abx for m19base
m19base1E1 =  np.array([[0.1816,0.1183,0.1061,0.1024,0.1022],[0.1902,0.1264 ,0.1175,0.1145,0.1167],
              [0.2025,0.1532,0.1567,0.1604,0.1659],[0.2129,0.1948,0.2233,0.2429,0.2484],
              [0.2072,0.1950,0.2189,0.2491,0.2546]])

m19base2E1 = np.array([[],[],
              [],[],
              []])

m19base3E1 = np.array([[0.1409,0.066,0.0643, 0.0634, 0.0616],[0.1383, 0.0626, 0.0588, 0.0587,0.0560],
              [0.1361,0.0594,0.0562,0.0554,0.0524],[0.1396,0.0575,0.0537,0.0534,0.0501],
              [0.1467,0.0603,0.0547,0.0537,0.0503]])


m19base4E1 = np.array([[],[],
              [],[],
              []])

title = 'ABX-error for the light models at different epochs during training'
fig = plt.figure(figsize=(15, 10))
fig.suptitle(title, fontsize=20)
plt.subplot(2, 2, 1)
plt.plot(m19base1E1.T[:, 0], label='layer1')
plt.plot(m19base1E1.T[:, 1], label='layer2')
plt.plot(m19base1E1.T[:, 2], label='layer3')
plt.plot(m19base1E1.T[:, 3], label='layer4')
plt.plot(m19base1E1.T[:, 4], label='layer5')
plt.title('w2v2',size=14)
#plt.xlabel('epoch', size=14)
plt.ylabel('abx-error', size=18)
#plt.xlim(-0.5, 4.5)
#plt.ylim(0, 0.25)
plt.xticks([0,1,2,3,4],['1', '5', '10', '15', '20'])
plt.grid()
plt.legend(fontsize=14)

plt.subplot(2, 2, 3)
plt.plot(m19base3E1.T[:, 0], label='layer1')
plt.plot(m19base3E1.T[:, 1], label='layer2')
plt.plot(m19base3E1.T[:, 2], label='layer3')
plt.plot(m19base3E1.T[:, 3], label='layer4')
plt.plot(m19base3E1.T[:, 4], label='layer5')
plt.title('VGS+', size=14)
plt.xlabel('epoch', size=14)
plt.ylabel('abx-error', size=18)
#plt.xlim(-0.5, 4.5)
#plt.ylim(0, 0.15)
plt.xticks([0,1,2,3,4],['1', '5', '10', '15', '20'])
plt.grid()
plt.legend(fontsize=14)
plt.savefig(os.path.join(path_save, 'fig2_light19_abx' + '.png'), format='png')
