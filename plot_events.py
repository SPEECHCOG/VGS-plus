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
path_event_6b = 'model6b/events.out.tfevents.1665144281.r13g01.bullx.830503.0'
path_event_6bTrim = 'model6bTrim/events.out.tfevents.1665332229.r13g02.bullx.1144814.0'
path_event_10bTrim = 'model10bTrim/events.out.tfevents.1665346196.r03g01.bullx.1369259.0'
path_event_18b = 'model18b/events.out.tfevents.1665249277.r03g02.bullx.465911.0'

# narvi

path_event_18bTrim = 'model18bTrim/events.out.tfevents.1665232396.nag14.tcsc-local.10609.0'
path_event_19bTrim = 'model19bTrim/events.out.tfevents.1665252989.nag12.tcsc-local.154664.0'


c_3 = 'blue'
c_4 = 'darkorange'
c_5 = 'green'
c_6 = 'red'
c_7 = 'royalblue'
c_18 = 'grey'
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
    return x_recall, y_recall

def find_single_lr (event, n):
    lr = pd.DataFrame(event.Scalars('lr'))
    x_lr = [ i/n for i in lr['step'][::100] ]
    y_lr = lr['value'][::100]
    return x_lr, y_lr

def find_single_vgsloss (event, n):
    vgsloss = pd.DataFrame(event.Scalars('coarse_matching_loss'))
    x_vgsloss = [ i/n for i in vgsloss['step'][::100] ]
    y_vgsloss = vgsloss['value'][::100]
    return x_vgsloss, y_vgsloss
 
def find_single_caploss (event, n):
    caploss = pd.DataFrame(event.Scalars('caption_w2v2_loss'))
    x_caploss = [ i/n for i in caploss['step'][::100] ]
    y_caploss = caploss['value'][::100]
    return x_caploss, y_caploss
   
def plot_single_event(event,n , name , title):

    x_recall, y_recall = find_single_recall (event, n)
    x_lr, y_lr = find_single_lr (event, n)
    x_vgsloss, y_vgsloss = find_single_vgsloss (event, n)
    x_caploss, y_caploss = find_single_caploss (event, n)
    fig = plt.figure()
    fig.suptitle(title)
    plt.subplot(1,2,1)
    plt.plot(x_recall,y_recall, label = 'recall@10')
    #plt.plot(x_lr,y_lr, label = 'lr')
    plt.xlabel('epoch')
    plt.grid()
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(x_vgsloss  , y_vgsloss, label = 'loss vgs')
    plt.plot(x_caploss  , y_caploss, 'olivedrab' ,label = 'loss caption')
    plt.xlabel('epoch')
    plt.grid()
    plt.legend()

    plt.savefig(os.path.join(path_save , name + '.png'), format = 'png')


###############################################################################

# event 6a
event_6a =  EventAccumulator(os.path.join(path_source, path_event_6a))
event_6a.Reload()
plot_single_event(event_6a , n_32 , 'model6a' , 'VGS+, random init, bs = 32' )    

# event 6b
event_6b =  EventAccumulator(os.path.join(path_source, path_event_6b))
event_6b.Reload()
plot_single_event(event_6b , n_64 , 'model6b' , 'VGS+, random init, bs = 64' )  


# event 6bTrim
event_6bTrim =  EventAccumulator(os.path.join(path_source, path_event_6bTrim))
event_6bTrim.Reload()
plot_single_event(event_6bTrim , n_64 , 'model6bTrim' , 'VGS+ (Trim-mask), random init, bs = 64' )  

# event 10bTrim
event_10bTrim =  EventAccumulator(os.path.join(path_source, path_event_10bTrim))
event_10bTrim.Reload()
plot_single_event(event_10bTrim , n_64 , 'model10bTrim' , 'VGS+ (Trim-mask), model 10b' ) 


# event 18b
event_18b =  EventAccumulator(os.path.join(path_source, path_event_18b))
event_18b.Reload()
plot_single_event(event_18b , n_64 , 'model18b' , 'VGS+lightlight, model 18b' ) 

# event 18bTrim
event_18bTrim =  EventAccumulator(os.path.join(path_source, path_event_18bTrim))
event_18bTrim.Reload()
plot_single_event(event_18bTrim , n_64 , 'model18bTrim' , 'VGS+lightlight(Trim-mask), model 18b' ) 

# event 19bTrim
event_19bTrim =  EventAccumulator(os.path.join(path_source, path_event_19bTrim))
event_19bTrim.Reload()
plot_single_event(event_19bTrim , n_64 , 'model19bTrim' , 'VGS+light (Trim-mask), model 19b' ) 

###########################
