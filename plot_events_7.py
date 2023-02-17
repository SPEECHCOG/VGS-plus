import os

import torch

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np

path_source = '/worktmp2/hxkhkh/current/FaST/experiments/'
path_save = '/worktmp2/hxkhkh/current/FaST/experiments/plots/eusipco/'

path_event_7base1T = 'model7base1T/exp'
path_event_7base2T = 'model7base2T/exp'
path_event_7base3 = 'model7base3/exp'


path_event_7ver4 = 'model7ver4/exp/'
path_event_7ver5 = 'model7ver5/exp/'
path_event_7ver6 = 'model7ver6/exp/'
path_event_7ver7 = 'model7ver7/exp/'
path_event_7ver8 = 'model7ver8/exp/'
#path_event_7ver9 = 'model7ver9/exp/'

# path_event_7ver14 = 'model7ver14/exp/'
# path_event_7ver26 = 'model7ver26/exp/'

c_1 = 'blue'
c_2 = 'green'
c_3 = 'orange'
c_4 = 'red'
c_5 = 'brown'
c_6 = 'darkorange'
c_7 = 'pink'
c_8 = 'royalblue'

n_32 = 18505
n_64 = 9252

label1 = 'W2V2'
label2 = 'VGS'
label3 = 'VGS+'
label4 = '(W2V2, VGS+)'
label5 = '(VGS, VGS+)'
label6 = '(W2V2, VGS)'
label7 = '(VGS, W2V2)'
label8 = '(VGS+, W2V2)'
label14 = '(W2V2-35E, VGS+)'
label26 = '(W2V2-5E, VGS)'


def find_average_train_load_timec(event):
    train_time = pd.DataFrame(event.Scalars('train_time'))['value']
    data_time = pd.DataFrame(event.Scalars('data_time'))['value']
    return train_time.mean , data_time.mean

def find_single_recall (event, n):
    recall = pd.DataFrame(event.Scalars('acc_r10'))   
    x_recall = [i/n for i in recall['step']]
    y_recall = recall['value'].to_list()    
    return x_recall, y_recall

def find_single_lr (event, n , interval):
    lr = pd.DataFrame(event.Scalars('lr'))
    x_lr = [ i/n for i in lr['step'][::interval] ]
    y_lr = lr['value'][::interval]
    return x_lr, y_lr

def find_single_vgsloss (event, n, interval):
    vgsloss = pd.DataFrame(event.Scalars('coarse_matching_loss'))
    x_vgsloss = [ i/n for i in vgsloss['step']]#[::interval] ]
    y_vgsloss = vgsloss['value']#[::interval]
    y_vgsloss_list = y_vgsloss.to_list()
    return x_vgsloss, y_vgsloss_list
 
def find_single_caploss (event, n , interval):
    caploss = pd.DataFrame(event.Scalars('caption_w2v2_loss'))
    x_caploss = [ i/n for i in caploss['step']]#[::interval] ]
    y_caploss = caploss['value']#[::interval]
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

############################################################################## Model 7 bases

event_7base1T =  EventAccumulator(os.path.join(path_source, path_event_7base1T))
event_7base1T.Reload()

event_7base2T =  EventAccumulator(os.path.join(path_source, path_event_7base2T))
event_7base2T.Reload()

event_7base3T =  EventAccumulator(os.path.join(path_source, path_event_7base3))
event_7base3T.Reload()

############################################################################## Model 7 versions

event_7ver4 =  EventAccumulator(os.path.join(path_source, path_event_7ver4))
event_7ver4.Reload()

event_7ver6 =  EventAccumulator(os.path.join(path_source, path_event_7ver6))
event_7ver6.Reload()


event_7ver5 =  EventAccumulator(os.path.join(path_source, path_event_7ver5))
event_7ver5.Reload()

event_7ver7 =  EventAccumulator(os.path.join(path_source, path_event_7ver7))
event_7ver7.Reload()

event_7ver8 =  EventAccumulator(os.path.join(path_source, path_event_7ver8))
event_7ver8.Reload()


# event_7ver14 =  EventAccumulator(os.path.join(path_source, path_event_7ver14))
# event_7ver14.Reload()

# event_7ver26 =  EventAccumulator(os.path.join(path_source, path_event_7ver26))
# event_7ver26.Reload()

kh
################################################################# Recalls
x_7base2T_recall, y_7base2T_recall = find_single_recall(event_7base2T, n_64)
x_7base3T_recall, y_7base3T_recall = find_single_recall(event_7base3T, n_64)


x_7ver4_recall, y_7ver4_recall = find_single_recall(event_7ver4, n_64)    
x_7ver6_recall, y_7ver6_recall = find_single_recall(event_7ver6, n_64)
 
x_7ver5_recall, y_7ver5_recall = find_single_recall(event_7ver5, n_64)

x_7ver8_recall, y_7ver8_recall = find_single_recall(event_7ver8, n_64)
    

# x_7ver14_recall, y_7ver14_recall = find_single_recall(event_7ver14, n_64) 
# x_7ver26_recall, y_7ver26_recall = find_single_recall(event_7ver26, n_64) 
   
x_recall_0 = 0
y_recall_0 = 0.002

x_7base2T_recall.insert(0, x_recall_0)
y_7base2T_recall.insert(0, y_recall_0)

x_7base3T_recall.insert(0, x_recall_0)
y_7base3T_recall.insert(0, y_recall_0)

x_7ver4_recall.insert(0, x_recall_0)
y_7ver4_recall.insert(0, y_recall_0)

x_7ver6_recall.insert(0, x_recall_0)
y_7ver6_recall.insert(0, y_recall_0)    

x_7ver5_recall.insert(0, x_recall_0)
y_7ver5_recall.insert(0, y_7base2T_recall [4])  

x_7ver8_recall.insert(0, x_recall_0)
y_7ver8_recall.insert(0, y_7base3T_recall [5])  
# x_7ver4_recall.insert(0, x_recall_0)
# y_7ver4_recall.insert(0, y_7base2T_recall [4])  
    
############ plot recalls for version 4, 6



fig = plt.figure(figsize=(7,10))

title = 'original versus pretrained models '
plt.subplot(2,1,1)

plt.plot(x_7base2T_recall, y_7base2T_recall, c_2, label = label2)
plt.plot(x_7base3T_recall, y_7base3T_recall, c_1, label = label3)
plt.plot(x_7ver4_recall, y_7ver4_recall, c_4, label = label4)
plt.plot(x_7ver5_recall, y_7ver5_recall, c_5, label = label5)
plt.plot(x_7ver6_recall, y_7ver6_recall, c_6, label = label6)

plt.plot(x_7ver8_recall, y_7ver8_recall, c_7, label = label8)

#plt.xlabel('epoch',size=18)
plt.ylabel('recall@10',size=18)
plt.ylim(0,1)
plt.grid()
plt.legend(fontsize=12)
plt.title(title)


title = ' pretrained models '
plt.subplot(2,1,2)
plt.plot(x_7ver4_recall, y_7ver4_recall, c_3, label = label3)
plt.plot(x_7ver6_recall, y_7ver6_recall, c_4, label = label4)

# plt.plot(x_7ver14_recall, y_7ver14_recall, c_6, label = label14)
# plt.plot(x_7ver26_recall, y_7ver26_recall, c_7, label = label26)

plt.xlabel('epoch',size=18)
plt.ylabel('recall@10',size=18)
plt.ylim(0,1)
plt.grid()
plt.legend(fontsize=12)
plt.title(title)
kh
#plt.savefig(os.path.join(path_save , 'model7_recall_versions' + '.png'), format = 'png')

################################################################# VGS loss
i = 500
def smooth_losses (x_in, y_in):
    n_epochs = int (x_in [-1])      
    x_out = np.arange(n_epochs)
    
    if 10000%n_epochs != 0:
       end =  10000 - 10000%n_epochs
    else:
       end = 10000
    y_out = y_in [0:end]   
    y_out = np.array(y_out)
    y_out = y_out.reshape(n_epochs,-1)
    y_out = np.mean(y_out, axis = 1)
    return x_out , y_out

x_7base1T_vgsloss, y_7base1T_vgsloss = find_single_vgsloss(event_7base1T, n_64 , i)
x_7base2T_vgsloss, y_7base2T_vgsloss = find_single_vgsloss(event_7base2T, n_64 , i)
x_7base3T_vgsloss, y_7base3T_vgsloss = find_single_vgsloss(event_7base3T, n_64, i)
x_7ver4_vgsloss, y_7ver4_vgsloss = find_single_vgsloss(event_7ver4, n_64, i)
x_7ver5_vgsloss, y_7ver5_vgsloss = find_single_vgsloss(event_7ver5, n_64, i)
x_7ver6_vgsloss, y_7ver6_vgsloss = find_single_vgsloss(event_7ver6, n_64, i)
x_7ver7_vgsloss, y_7ver7_vgsloss = find_single_vgsloss(event_7ver7, n_64, i)
x_7ver8_vgsloss, y_7ver8_vgsloss = find_single_vgsloss(event_7ver8, n_64, i)


x_7base1T_vgsloss, y_7base1T_vgsloss = smooth_losses (x_7base1T_vgsloss, y_7base1T_vgsloss)
x_7base2T_vgsloss, y_7base2T_vgsloss = smooth_losses (x_7base2T_vgsloss, y_7base2T_vgsloss)
x_7base3T_vgsloss, y_7base3T_vgsloss = smooth_losses (x_7base3T_vgsloss, y_7base3T_vgsloss)
x_7ver4_vgsloss, y_7ver4_vgsloss = smooth_losses(x_7ver4_vgsloss, y_7ver4_vgsloss )
x_7ver5_vgsloss, y_7ver5_vgsloss = smooth_losses(x_7ver5_vgsloss, y_7ver5_vgsloss )
x_7ver6_vgsloss, y_7ver6_vgsloss = smooth_losses(x_7ver6_vgsloss, y_7ver6_vgsloss )
x_7ver7_vgsloss, y_7ver7_vgsloss = smooth_losses(x_7ver7_vgsloss, y_7ver7_vgsloss )
x_7ver8_vgsloss, y_7ver8_vgsloss = smooth_losses(x_7ver8_vgsloss, y_7ver8_vgsloss )

################################################################# caption loss
i = 500
x_7base1T_caploss, y_7base1T_caploss = find_single_caploss(event_7base1T, n_64 , i)
x_7base2T_caploss, y_7base2T_caploss = find_single_caploss(event_7base2T, n_64 , i)
x_7base3T_caploss, y_7base3T_caploss = find_single_caploss(event_7base3T, n_64, i)

x_7ver4_caploss, y_7ver4_caploss = find_single_caploss(event_7ver4, n_64, i)
x_7ver5_caploss, y_7ver5_caploss = find_single_caploss(event_7ver5, n_64, i)
x_7ver6_caploss, y_7ver6_caploss = find_single_caploss(event_7ver6, n_64, i)
x_7ver7_caploss, y_7ver7_caploss = find_single_caploss(event_7ver7, n_64, i)
x_7ver8_caploss, y_7ver8_caploss = find_single_caploss(event_7ver8, n_64, i)

x_7base1T_caploss, y_7base1T_caploss = smooth_losses (x_7base1T_caploss, y_7base1T_caploss)
x_7base2T_caploss, y_7base2T_caploss = smooth_losses (x_7base2T_caploss, y_7base2T_caploss)
x_7base3T_caploss, y_7base3T_caploss = smooth_losses (x_7base3T_caploss, y_7base3T_caploss)
x_7ver4_caploss, y_7ver4_caploss = smooth_losses(x_7ver4_caploss, y_7ver4_caploss)
x_7ver5_caploss, y_7ver5_caploss = smooth_losses(x_7ver5_caploss, y_7ver5_caploss)
x_7ver6_caploss, y_7ver6_caploss = smooth_losses(x_7ver6_caploss, y_7ver6_caploss)
x_7ver7_caploss, y_7ver7_caploss = smooth_losses(x_7ver7_caploss, y_7ver7_caploss)
x_7ver8_caploss, y_7ver8_caploss = smooth_losses(x_7ver8_caploss, y_7ver8_caploss)
kh
#################################################################
#################################################################

#............. plotting losses for base models in single plot

#################################################################
################################################################# 
x_pre = np.arange(-20,0)
fig = plt.figure(figsize=(7,21))
fsize = 16
ax = fig.add_subplot(3, 1, 1)#plt.subplot(2,2,1)
plt.plot(x_7base1T_vgsloss+1, y_7base1T_vgsloss, c_1, label = 'AV, VGS ')
plt.plot(x_7base2T_vgsloss+1, y_7base2T_vgsloss, c_2, label = 'AV, VGS ')
plt.plot(x_7base3T_vgsloss+1, y_7base3T_vgsloss, c_3, label = 'AV, VGS+ ')
plt.plot(x_7base1T_caploss+1, y_7base1T_caploss, c_1, linestyle='dashed', label = 'AUD, W2V2 ')
#plt.plot(x_7base2T_caploss+1, y_7base2T_caploss, c_2, label = 'AUD, W2V2 ')
plt.plot(x_7base3T_caploss+1, y_7base3T_caploss, c_3, linestyle='dashed', label = 'AUD, VGS+ ')
ax.set_yscale('log')
plt.xticks(ticks = np.arange(0,75,10),size=fsize+2 )
plt.yticks([0.1,1,10],['0.1','1','10'],size=fsize+2)
#plt.xlabel('Epoch',size=fsize)
plt.ylabel('Training loss',size=fsize+4)
plt.grid()
plt.legend(fontsize=fsize)
#plt.savefig(os.path.join(path_save , 'losses_base' + '.png'), format = 'png')
############################
#fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(3, 1, 2) #plt.subplot(2,2,3)
plt.plot(x_7base1T_vgsloss+1, y_7base1T_vgsloss, c_1, label = 'W2V2 ')
plt.plot(x_7base2T_vgsloss+1, y_7base2T_vgsloss, c_2, label = 'VGS ')
plt.plot(x_7base3T_vgsloss+1, y_7base3T_vgsloss, c_3, label = 'VGS+')
plt.plot(x_7ver4_vgsloss+1, y_7ver4_vgsloss, c_4, label = label4)
plt.plot(x_7ver5_vgsloss+1, y_7ver5_vgsloss, c_5, label = label5)
plt.plot(x_7ver6_vgsloss+1, y_7ver6_vgsloss, c_6, label = label6)
plt.plot(x_7ver7_vgsloss+1, y_7ver7_vgsloss, c_7, label = label7)
plt.plot(x_7ver8_vgsloss+1, y_7ver8_vgsloss, c_8, label = label8)
ax.set_yscale('log')
plt.xticks(ticks = np.arange(0,75,10),size=fsize+2 )
plt.yticks([0.1,1,10],['0.1','1','10'],size=fsize+2)
#plt.xlabel('Epoch',size=fsize)
plt.ylabel('Training loss_AV',size=fsize+4)
plt.grid()
plt.legend(fontsize=fsize)
#plt.savefig(os.path.join(path_save , 'lossVG_versions' + '.png'), format = 'png')
############################
#fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(3, 1, 3)#plt.subplot(2,2,4)
plt.plot(x_7base1T_caploss+1, y_7base1T_caploss, c_1, label = 'W2V2')
plt.plot(x_7base3T_caploss+1, y_7base3T_caploss, c_3, label = 'VGS+')
plt.plot(x_7ver4_caploss+1, y_7ver4_caploss, c_4, label = label4)
plt.plot(x_7ver5_caploss[1:], y_7ver5_caploss[1:], c_5, label = label5)
plt.plot(x_7ver6_caploss[1:], y_7ver6_caploss[1:], c_6, label = label6)
plt.plot(x_7ver7_caploss[1:], y_7ver7_caploss[1:], c_7, label = label7)
plt.plot(x_7ver8_caploss+1, y_7ver8_caploss, c_8, label = label8)
ax.set_yscale('log')
plt.xticks(ticks = np.arange(0,75,10),size=fsize+2 )
plt.yticks([1,5, 10],['1','5','10'],size=fsize+2)
plt.xlabel('\nEpoch',size=fsize+4)
plt.ylabel('Training loss_AUD',size=fsize+4)
plt.grid()
plt.legend(fontsize=fsize)
#plt.savefig(os.path.join(path_save , 'lossAUD_versions' + '.png'), format = 'png')
plt.savefig(os.path.join(path_save , 'loss-log' + '.pdf'), format = 'pdf', bbox_inches='tight')

kh


#################################################################
#################################################################

#............. plotting losses with pretrained ones

#################################################################
################################################################# 
x_pre = np.arange(1,21)
fig = plt.figure(figsize=(7,21))
fsize = 16
ax = fig.add_subplot(3, 1, 1)#plt.subplot(2,2,1)
plt.plot(x_7base1T_vgsloss+1, y_7base1T_vgsloss, c_1, label = 'AV, VGS ')
plt.plot(x_7base2T_vgsloss+1, y_7base2T_vgsloss, c_2, label = 'AV, VGS ')
plt.plot(x_7base3T_vgsloss+1, y_7base3T_vgsloss, c_3, label = 'AV, VGS+ ')
plt.plot(x_7base1T_caploss+1, y_7base1T_caploss, c_1, linestyle='dashed', label = 'AUD, W2V2 ')
#plt.plot(x_7base2T_caploss+1, y_7base2T_caploss, c_2, label = 'AUD, W2V2 ')
plt.plot(x_7base3T_caploss+1, y_7base3T_caploss, c_3, linestyle='dashed', label = 'AUD, VGS+ ')
ax.set_yscale('log')
plt.xticks(ticks = np.arange(0,75,10),size=fsize+2 )
plt.yticks([0.1,1,10],['0.1','1','10'],size=fsize+2)
#plt.xlabel('Epoch',size=fsize)
plt.ylabel('Training loss',size=fsize+4)
plt.grid()
plt.legend(fontsize=fsize)
#plt.savefig(os.path.join(path_save , 'losses_base' + '.png'), format = 'png')
############################
x_pre = np.arange(1,21)
fig = plt.figure(figsize=(14,7))
fsize = 16
LW = 5
plt.subplot(1,2,1) #ax = fig.add_subplot(3, 1, 2) #plt.subplot(2,2,3)
plt.plot(x_7base1T_vgsloss+1, y_7base1T_vgsloss, c_1, label = 'W2V2', lw=LW)
plt.plot(x_7base2T_vgsloss+1, y_7base2T_vgsloss, c_2, label = 'VGS', lw=LW)
plt.plot(x_7base3T_vgsloss+1, y_7base3T_vgsloss, c_3, label = 'VGS+', lw=LW)
plt.plot(np.concatenate((x_pre, x_7ver4_vgsloss+21)), np.concatenate((y_7base1T_vgsloss[0:20], y_7ver4_vgsloss)), c_1, label = label4, linestyle='dashed', lw=LW-2)
plt.plot(np.concatenate((x_pre, x_7ver5_vgsloss+21)), np.concatenate((y_7base2T_vgsloss[0:20], y_7ver5_vgsloss)), c_2, label = label5, linestyle='dashed', lw=LW-2)
plt.plot(np.concatenate((x_pre, x_7ver6_vgsloss+21)), np.concatenate((y_7base1T_vgsloss[0:20], y_7ver6_vgsloss)), c_1, label = label6, linestyle='dotted', lw=LW)
plt.plot(np.concatenate((x_pre, x_7ver7_vgsloss+21)), np.concatenate((y_7base2T_vgsloss[0:20], y_7ver7_vgsloss)), c_2, label = label7, linestyle='dotted', lw=LW)
plt.plot(np.concatenate((x_pre,x_7ver8_vgsloss+21)), np.concatenate((y_7base3T_vgsloss[0:20], y_7ver8_vgsloss)), c_3, label = label8, linestyle='dashed', lw=LW-2)
plt.yscale('log')#ax.set_yscale('log')
plt.xticks(ticks = np.arange(0,75,10),size=fsize+2 )
plt.yticks([0.1, 0.2, 0.5 , 1,10],['0.1','0.2', '0.5' ,'1','10'],size=fsize+2)
plt.xlabel('\nEpoch',size=fsize+2)
plt.ylabel('Loss_AV',size=fsize+3)
plt.ylim( [0.09,20] )
plt.grid()
############################
plt.subplot(1,2,2) #ax = fig.add_subplot(3, 1, 3)#plt.subplot(2,2,4)
plt.plot(x_7base1T_caploss+1, y_7base1T_caploss, c_1, label = 'W2V2', lw=LW)
plt.plot(x_7base2T_caploss+1, y_7base2T_caploss, c_2, label = 'VGS', lw=LW)
plt.plot(x_7base3T_caploss+1, y_7base3T_caploss, c_3, label = 'VGS+', lw=LW)
plt.plot(np.concatenate((x_pre, x_7ver4_caploss+21)), np.concatenate(( y_7base1T_caploss[0:20], y_7ver4_caploss)), c_1, label = label4, linestyle='dashed', lw=LW-2)
plt.plot(np.concatenate((x_pre, x_7ver5_caploss+21)), np.concatenate(( y_7base2T_caploss[0:20], y_7ver5_caploss)), c_2, label = label5, linestyle='dashed', lw=LW-2)
plt.plot(np.concatenate((x_pre, x_7ver6_caploss+21)), np.concatenate(( y_7base1T_caploss[0:20], y_7ver6_caploss)), c_1, label = label6, linestyle='dotted', lw=LW)
plt.plot(np.concatenate((x_pre, x_7ver7_caploss+21)), np.concatenate(( y_7base2T_caploss[0:20], y_7ver7_caploss)), c_2, label = label7, linestyle='dotted', lw=LW)
plt.plot(np.concatenate((x_pre, x_7ver8_caploss+21)), np.concatenate(( y_7base3T_caploss[0:20], y_7ver8_caploss)), c_3, label = label8, linestyle='dashed', lw=LW-2)
plt.yscale('log')
plt.xticks(ticks = np.arange(0,75,10),size=fsize+2 )
plt.yticks([ 1,2,3,4,5,6,10],['1','2','3','4','5','','10'],size=fsize+2)
plt.xlabel('\nEpoch',size=fsize+2)
plt.ylabel('Loss_AUD',size=fsize+3)
plt.ylim( [0.9,12] )
plt.grid()
plt.legend(fontsize=fsize , bbox_to_anchor=(0.6, 0.45)) # (1.4, 1.2) (0.09, 1.5)(1., 1.)
#plt.savefig(os.path.join(path_save , 'lossAUD_versions' + '.png'), format = 'png')
plt.savefig(os.path.join(path_save , 'loss-log-2c' + '.pdf'), format = 'pdf', bbox_inches='tight')
#################################################################
#################################################################
################################################################# plotting losses for base models separately
label1 = 'w2v2'
label2 = 'VGS'
label3 = 'VGS+'


fig = plt.figure(figsize=(10,10))
title = ' vgs loss '
plt.subplot(2,2,1)
plt.plot(x_7base2T_vgsloss, y_7base2T_vgsloss, c_2, label = label2)
plt.plot(x_7base3T_vgsloss, y_7base3T_vgsloss, c_3, label = label3)
#plt.xlabel('epoch',size=18)
#plt.ylabel('vgs loss',size=18)
plt.ylim(0,10.5)
plt.xticks(ticks = np.arange(0,50,5) )
plt.yticks(ticks = np.arange(0,11) )
plt.grid()
plt.legend(fontsize=12)
plt.title(title)

title = ' caption loss '
plt.subplot(2,2,2)
plt.plot(x_7base1T_caploss, y_7base1T_caploss, c_1, label = label1)
plt.plot(x_7base3T_caploss, y_7base3T_caploss, c_3, label = label3)
#plt.ylabel('caption loss',size=18)
plt.ylim(0,10.5)
plt.xticks(ticks = np.arange(0,50,5) )
plt.yticks(ticks = np.arange(0,11) )
plt.grid()
plt.legend(fontsize=12)
plt.title(title)

title = ' VGS+ '
plt.subplot(2,2,3)
plt.plot(x_7base3T_vgsloss, y_7base3T_vgsloss, c_3, label = 'vgs loss')
plt.plot(x_7base3T_caploss, y_7base3T_caploss, c_3, linestyle='dashed' ,label = 'caaption loss')
#plt.ylabel('caption loss',size=18)
plt.xlabel('epoch',size=18)
plt.ylim(0,10.5)
plt.xticks(ticks = np.arange(0,50,5) )
plt.yticks(ticks = np.arange(0,11) )
plt.grid()
plt.legend(fontsize=12)
plt.title(title)

plt.savefig(os.path.join(path_save , 'model7_losses_original' + '.png'), format = 'png')

################################################################# plotting losses for versions


fig = plt.figure(figsize=(10,10))

title = ' vgs loss '
plt.subplot(2,2,1)
plt.plot(x_7base2T_vgsloss, y_7base2T_vgsloss, c_2, label = 'VGS')
plt.plot(x_7base3T_vgsloss, y_7base3T_vgsloss, c_3, label = 'VGS+')
plt.ylim(0,10.5)
plt.xticks(ticks = np.arange(0,50,5) )
plt.yticks(ticks = np.arange(0,11) )
plt.grid()
plt.legend(fontsize=12)
plt.title(title)

title = ' vgs loss '
plt.subplot(2,2,2)
plt.plot(x_7base3T_vgsloss, y_7base3T_vgsloss, c_3, label = 'VGS+')
plt.plot(x_7ver4_vgsloss, y_7ver4_vgsloss, c_4, label = label4)
plt.plot(x_7ver6_vgsloss, y_7ver6_vgsloss, c_6, label = label6)
plt.plot(x_7ver5_vgsloss, y_7ver5_vgsloss, c_5, label = label5)
plt.plot(x_7ver8_vgsloss, y_7ver8_vgsloss, c_8, label = label8)
plt.ylim(0,10.5)
plt.xticks(ticks = np.arange(0,50,5) )
plt.yticks(ticks = np.arange(0,11) )
plt.grid()
plt.legend(fontsize=12)
plt.title(title)

title = ' caption loss '
plt.subplot(2,2,3)
plt.plot(x_7base1T_caploss, y_7base1T_caploss, c_1, label = 'w2v2')
plt.plot(x_7base3T_caploss, y_7base3T_caploss, c_3, label = 'VGS+')
plt.plot(x_7ver4_caploss, y_7ver4_caploss, c_4, label = 'VGS+ pretrained w2v2')
plt.ylim(0,5.5)
plt.xticks(ticks = np.arange(0,50,5) )
plt.yticks(ticks = np.arange(0,5) )
plt.grid()
plt.legend(fontsize=12)
plt.title(title)

title = ' caption loss '
plt.subplot(2,2,4)
plt.plot(x_7base1T_caploss, y_7base1T_caploss, c_1, label = 'w2v2')
plt.plot(x_7base3T_caploss, y_7base3T_caploss, c_3, label = 'VGS+')
plt.plot(x_7ver4_caploss, y_7ver4_caploss, c_4, label = label4)
plt.plot(x_7ver5_caploss, y_7ver5_caploss, c_5, label = label5)
plt.plot(x_7ver7_caploss, y_7ver7_caploss, c_7, label = label7)
plt.plot(x_7ver8_caploss, y_7ver8_caploss, c_8, label = label8)
plt.ylim(0,5.5)
plt.xticks(ticks = np.arange(0,50,5) )
plt.yticks(ticks = np.arange(0,5) )
plt.grid()
plt.legend(fontsize=12)
plt.title(title)


plt.savefig(os.path.join(path_save , 'model7_losses_versions' + '.png'), format = 'png')
