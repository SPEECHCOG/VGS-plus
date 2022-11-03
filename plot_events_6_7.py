import os

import torch

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np

path_source = '/worktmp2/hxkhkh/current/FaST/experiments/'
path_save = '/worktmp2/hxkhkh/current/FaST/experiments/plots/'

path_event_6base1T = 'model6base1T/'
path_event_6base1F = 'model6base1F/'
path_event_6base2T = 'model6base2T/'
path_event_6base3T = 'model6base3T/'
path_event_6base4T = 'model6base4T/'

path_event_7base1T = 'model7base1T/'
path_event_7base1F = 'model7base1F/'
path_event_7base2T = 'model7base2T/'
path_event_7base3T = 'model7base3T/'
path_event_7base4T = 'model7base4T/'

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
    
    
############################################################################## Model 6
# event_6base1T =  EventAccumulator(os.path.join(path_source, path_event_6base1T))
# event_6base1T.Reload()

# event_6base2T =  EventAccumulator(os.path.join(path_source, path_event_6base2T))
# event_6base2T.Reload()

# event_6base3T =  EventAccumulator(os.path.join(path_source, path_event_6base3T))
# event_6base3T.Reload()
# ################################################################# Recalls
# x_6base2T_recall, y_6base2T_recall = find_single_recall(event_6base2T, n_64)
# x_6base3T_recall, y_6base3T_recall = find_single_recall(event_6base3T, n_64)

# ################################################################# VGS loss
# i = 500
# x_6base2T_vgsloss, y_6base2T_vgsloss = find_single_vgsloss(event_6base2T, n_64 , i)
# x_6base3T_vgsloss, y_6base3T_vgsloss = find_single_vgsloss(event_6base3T, n_64, i)

# ################################################################# caption loss
# i = 500
# x_6base1T_caploss, y_6base1T_caploss = find_single_caploss(event_6base1T, n_64 , i)
# x_6base3T_caploss, y_6base3T_caploss = find_single_caploss(event_6base3T, n_64, i)

# ################################################################# plotting recall and loss for model 6
# title = 'model origin, with gradmul = 0.1'
# label1 = 'w2v2'
# label2 = 'VGS'
# label3 = 'VGS+'
# label4 = 'VGS+ pretrained'
# label5 = 'VGS+ reference'
# fig = plt.figure(figsize=(15,10))
# fig.suptitle(title, fontsize=20)
# # recall
# plt.subplot(1,3,1)
# plt.plot(x_6base2T_recall, y_6base2T_recall, c_2, label = label2)
# plt.plot(x_6base3T_recall, y_6base3T_recall, c_3, label = label3)
# plt.xlabel('epoch',size=18)
# plt.ylabel('recall@10',size=18)
# plt.ylim(0,0.8)
# plt.grid()
# plt.legend(fontsize=16)
# #vgs loss
# plt.subplot(1,3,2)
# plt.plot(x_6base2T_vgsloss, y_6base2T_vgsloss, c_2, label = label2)
# plt.plot(x_6base3T_vgsloss, y_6base3T_vgsloss, c_3, label = label3)
# plt.xlabel('epoch',size=18)
# plt.ylabel('VGS loss',size=18)
# plt.ylim(0,14)
# plt.grid()
# plt.legend(fontsize=16)
# # cap loss
# plt.subplot(1,3,3)
# plt.plot(x_6base1T_caploss, y_6base1T_caploss, c_1, label = label1)
# plt.plot(x_6base3T_caploss, y_6base3T_caploss, c_3, label = label3)
# plt.xlabel('epoch',size=18)
# plt.ylabel('lcaption loss',size=18)
# plt.ylim(0,14)
# plt.grid()
# plt.legend(fontsize=16)
# plt.savefig(os.path.join(path_save , 'model6_recall_loss' + '.png'), format = 'png')


############################################################################## Model 7
event_7base1T =  EventAccumulator(os.path.join(path_source, path_event_7base1T))
event_7base1T.Reload()

event_7base2T =  EventAccumulator(os.path.join(path_source, path_event_7base2T))
event_7base2T.Reload()

event_7base3T =  EventAccumulator(os.path.join(path_source, path_event_7base3T))
event_7base3T.Reload()
################################################################# Recalls
x_7base2T_recall, y_7base2T_recall = find_single_recall(event_7base2T, n_64)
x_7base3T_recall, y_7base3T_recall = find_single_recall(event_7base3T, n_64)

################################################################# VGS loss
i = 500
x_7base2T_vgsloss, y_7base2T_vgsloss = find_single_vgsloss(event_7base2T, n_64 , i)
x_7base3T_vgsloss, y_7base3T_vgsloss = find_single_vgsloss(event_7base3T, n_64, i)

################################################################# caption loss
i = 500
x_7base1T_caploss, y_7base1T_caploss = find_single_caploss(event_7base1T, n_64 , i)
x_7base3T_caploss, y_7base3T_caploss = find_single_caploss(event_7base3T, n_64, i)

################################################################# plotting recall and loss for model 6
title = 'model origin, with gradmul = 0.1'
label1 = 'w2v2'
label2 = 'VGS'
label3 = 'VGS+'
label4 = 'VGS+ pretrained'
label5 = 'VGS+ reference'
fig = plt.figure(figsize=(15,10))
fig.suptitle(title, fontsize=20)
# recall
plt.subplot(1,3,1)
plt.plot(x_7base2T_recall, y_7base2T_recall, c_2, label = label2)
plt.plot(x_7base3T_recall, y_7base3T_recall, c_3, label = label3)
plt.xlabel('epoch',size=18)
plt.ylabel('recall@10',size=18)
plt.ylim(0,0.8)
plt.grid()
plt.legend(fontsize=16)
#vgs loss
plt.subplot(1,3,2)
plt.plot(x_7base2T_vgsloss, y_7base2T_vgsloss, c_2, label = label2)
plt.plot(x_7base3T_vgsloss, y_7base3T_vgsloss, c_3, label = label3)
plt.xlabel('epoch',size=18)
plt.ylabel('VGS loss',size=18)
plt.ylim(0,14)
plt.grid()
plt.legend(fontsize=16)
# cap loss
plt.subplot(1,3,3)
plt.plot(x_7base1T_caploss, y_7base1T_caploss, c_1, label = label1)
plt.plot(x_7base3T_caploss, y_7base3T_caploss, c_3, label = label3)
plt.xlabel('epoch',size=18)
plt.ylabel('lcaption loss',size=18)
plt.ylim(0,14)
plt.grid()
plt.legend(fontsize=16)
plt.savefig(os.path.join(path_save , 'model6_recall_loss' + '.png'), format = 'png')
