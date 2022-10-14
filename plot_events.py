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
path_event_18bTrim = 'model18bTrim/events.out.tfevents.1665232396.nag14.tcsc-local.10609.0'
path_event_19bTrim = 'model19bTrim/events.out.tfevents.1665252989.nag12.tcsc-local.154664.0'

########################
path_event_19bT3 = 'model19bT3/events.out.tfevents.1665591025.nag08.tcsc-local.25712.0'
path_event_19bT4 = 'model19bT4/events.out.tfevents.1665440508.r14g01.bullx.2583427.0'
path_event_19bT5 = 'model19bT5/events.out.tfevents.1665701070.r13g02.bullx.2734031.0'
path_event_19bT6 = 'model19bT6/events.out.tfevents.1665693739.r18g04.bullx.2469046.0'
path_event_19bT7 = 'model19bT7/events.out.tfevents.1665634174.nag14.tcsc-local.25605.0'
path_event_19bT8 = 'model19bT8/events.out.tfevents.1665692696.nag12.tcsc-local.251956.0'


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
    return x_recall, y_recall

def plot_double_events(event1,event2,label1,label2,c1,c2, n, pltname, title):

    x1_recall, y1_recall = find_single_recall (event1, n)   
    x1_vgsloss, y1_vgsloss = find_single_vgsloss (event1, n)
    x1_caploss, y1_caploss = find_single_caploss (event1, n)
    
    x2_recall, y2_recall = find_single_recall (event2, n)   
    x2_vgsloss, y2_vgsloss = find_single_vgsloss (event2, n)
    x2_caploss, y2_caploss = find_single_caploss (event2, n)
    
    fig = plt.figure(figsize=(15,10))
    fig.suptitle(title, fontsize=18)
    plt.subplot(1,3,1)
    plt.plot(x1_recall,y1_recall, c1, label = label1)
    plt.plot(x2_recall,y2_recall, c2, label = label2)
    #plt.plot(x_lr,y_lr, label = 'lr')
    plt.xlabel('epoch',size=14)
    plt.ylabel('recall@10',size=14)
    plt.ylim(0,0.8)
    plt.grid()
    plt.legend(fontsize=14)
    plt.subplot(1,3,2)
    plt.plot(x1_vgsloss  , y1_vgsloss, c1, label = label1)
    plt.plot(x2_vgsloss  , y2_vgsloss, c2, label = label2)
    plt.xlabel('epoch',size=14)
    plt.ylabel('loss-vgs',size=14)
    plt.ylim(0,14)
    plt.grid()
    plt.legend(fontsize=14)
    plt.subplot(1,3,3)
    plt.plot(x1_caploss  , y1_caploss, c1, label = label1)
    plt.plot(x2_caploss  , y2_caploss, c2, label = label2)
    plt.xlabel('epoch',size=14)
    plt.ylabel('loss-caption',size=14)
    plt.ylim(0,14)
    plt.grid()
    plt.legend(fontsize=14)

    plt.savefig(os.path.join(path_save , pltname + '.png'), format = 'png')
    
    def plot_several_events(event1,event2,event3,event4,levent5,event6,abel1,label2,label3,label4,label5,label6, n, pltname, title):

        x1_recall, y1_recall = find_single_recall (event1, n)   
        x1_vgsloss, y1_vgsloss = find_single_vgsloss (event1, n)
        x1_caploss, y1_caploss = find_single_caploss (event1, n)
        
        x2_recall, y2_recall = find_single_recall (event2, n)   
        x2_vgsloss, y2_vgsloss = find_single_vgsloss (event2, n)
        x2_caploss, y2_caploss = find_single_caploss (event2, n)
        
        fig = plt.figure(figsize=(15,10))
        fig.suptitle(title, fontsize=18)
        plt.subplot(1,3,1)
        plt.plot(x1_recall,y1_recall, c1, label = label1)
        plt.plot(x2_recall,y2_recall, c2, label = label2)
        #plt.plot(x_lr,y_lr, label = 'lr')
        plt.xlabel('epoch',size=14)
        plt.ylabel('recall@10',size=14)
        plt.ylim(0,0.8)
        plt.grid()
        plt.legend(fontsize=14)
        plt.subplot(1,3,2)
        plt.plot(x1_vgsloss  , y1_vgsloss, c1, label = label1)
        plt.plot(x2_vgsloss  , y2_vgsloss, c2, label = label2)
        plt.xlabel('epoch',size=14)
        plt.ylabel('loss-vgs',size=14)
        plt.ylim(0,14)
        plt.grid()
        plt.legend(fontsize=14)
        plt.subplot(1,3,3)
        plt.plot(x1_caploss  , y1_caploss, c1, label = label1)
        plt.plot(x2_caploss  , y2_caploss, c2, label = label2)
        plt.xlabel('epoch',size=14)
        plt.ylabel('loss-caption',size=14)
        plt.ylim(0,14)
        plt.grid()
        plt.legend(fontsize=14)

        plt.savefig(os.path.join(path_save , pltname + '.png'), format = 'png')
###############################################################################
kh
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

######### Model 3
event_19bT3 =  EventAccumulator(os.path.join(path_source, path_event_19bT3))
event_19bT3.Reload()
x_recall, y_recall = plot_single_event(event_19bT3 , n_64 , 'm19bT3' , 'model 3, sinusoid function' ) 

plot_double_events(event_19bTrim, event_19bT3,'baseline','3','gray','green', n_64, 'm19bT3_19','m193 versus baseline')

######### Model 4
event_19bT4 =  EventAccumulator(os.path.join(path_source, path_event_19bT4))
event_19bT4.Reload()
x_recall, y_recall = plot_single_event(event_19bT4 , n_64 , 'm19bT4' , 'model 4, step function' ) 

plot_double_events(event_19bTrim, event_19bT4,'baseline','4','gray','green', n_64, 'm19bT4_19','m194 versus baseline')

######### Model 5
event_19bT5 =  EventAccumulator(os.path.join(path_source, path_event_19bT5))
event_19bT5.Reload()
x_recall, y_recall = plot_single_event(event_19bT5 , n_64 , 'm19bT5' , 'model 5, linear increasing' ) 

plot_double_events(event_19bTrim, event_19bT5,'baseline','5','gray','green', n_64, 'm19bT5_19','m195 versus baseline')

######### Model 6
event_19bT6 =  EventAccumulator(os.path.join(path_source, path_event_19bT6))
event_19bT6.Reload()
x_recall, y_recall = plot_single_event(event_19bT6 , n_64 , 'm19bT6' , 'model 6, linear decreasing' ) 

plot_double_events(event_19bTrim, event_19bT6,'baseline','6','gray','green', n_64, 'm19bT6_19','m196 versus baseline')

######### Model 7
event_19bT7 =  EventAccumulator(os.path.join(path_source, path_event_19bT7))
event_19bT7.Reload()
x_recall, y_recall = plot_single_event(event_19bT7 , n_64 , 'm19bT7' , 'model 7, alpha = 0.1' ) 

plot_double_events(event_19bTrim, event_19bT7,'baseline','7','gray','green', n_64, 'm19bT7_19','m197 versus baseline')

######### Model 8

event_19bT8 =  EventAccumulator(os.path.join(path_source, path_event_19bT8))
event_19bT8.Reload()
x_recall, y_recall = plot_single_event(event_19bT8 , n_64 , 'm19bT8' , 'model 8,  alpha = 0.9' ) 

plot_double_events(event_19bTrim, event_19bT8,'baseline','8','gray','green', n_64, 'm19bT8_19','m198 versus baseline')










