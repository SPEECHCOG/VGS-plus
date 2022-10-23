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
path_event_6bTrim = 'model6bTrim/events.out.tfevents.1665332229.r13g02.bullx.1144814.0'

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
    
    def plot_several_events(event1,event2,event3,event4,levent5,event6,label1,label2,label3,label4,label5,label6, n, pltname, title):

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

######### Model 0
event_19bT0 =  EventAccumulator(os.path.join(path_source, path_event_19bT0))
event_19bT0.Reload()
x_recall, y_recall = plot_single_event(event_19bT0 , n_64 , 'm19bT3' , 'model 0, sinusoid function' ) 

plot_double_events(event_19base3, event_19bT0, 'baseline ','19bT0 ','gray','green', n_64, 'm19bT0_19','19bT0 (alpha = 0.001) versus baseline')

######### Model 3
event_19bT3 =  EventAccumulator(os.path.join(path_source, path_event_19bT3))
event_19bT3.Reload()
x_recall, y_recall = plot_single_event(event_19bT3 , n_64 , 'm19bT3' , 'model 3, sinusoid function' ) 

plot_double_events(event_19base3, event_19bT3, 'baseline ','19bT3 ','gray','green', n_64, 'm19bT3_19','19bT3 versus baseline')

######### Model 4
event_19bT4 =  EventAccumulator(os.path.join(path_source, path_event_19bT4))
event_19bT4.Reload()
x_recall, y_recall = plot_single_event(event_19bT4 , n_64 , 'm19bT4' , 'model 4, step function' ) 

plot_double_events(event_19base3, event_19bT4,'baseline ','19bT4 ','gray','green', n_64, 'm19bT4_19','19bT4 versus baseline')

######### Model 5
event_19bT5 =  EventAccumulator(os.path.join(path_source, path_event_19bT5))
event_19bT5.Reload()
x_recall, y_recall = plot_single_event(event_19bT5 , n_64 , 'm19bT5' , 'model 5, linear increasing' ) 

plot_double_events(event_19base3, event_19bT5,'baseline ','19bT5 ','gray','green', n_64, 'm19bT5_19','19bT5 versus baseline')

######### Model 6
event_19bT6 =  EventAccumulator(os.path.join(path_source, path_event_19bT6))
event_19bT6.Reload()
x_recall, y_recall = plot_single_event(event_19bT6 , n_64 , 'm19bT6' , 'model 6, linear decreasing' ) 

plot_double_events(event_19base3, event_19bT6,'baseline ','19bT6 ','gray','green', n_64, 'm19bT6_19','19bT6 versus baseline')

######### Model 7
event_19bT7 =  EventAccumulator(os.path.join(path_source, path_event_19bT7))
event_19bT7.Reload()
x_recall, y_recall = plot_single_event(event_19bT7 , n_64 , 'm19bT7' , 'model 7, alpha = 0.1' ) 

plot_double_events(event_19base3, event_19bT7,'baseline ','19bT7 ','gray','green', n_64, 'm19bT7_19','19bT7 versus baseline')

######### Model 8

event_19bT8 =  EventAccumulator(os.path.join(path_source, path_event_19bT8))
event_19bT8.Reload()
x_recall, y_recall = plot_single_event(event_19bT8 , n_64 , 'm19bT8' , 'model 8,  alpha = 0.9' ) 

plot_double_events(event_19base3, event_19bT8,'baseline ','19bT8 ','gray','green', n_64, 'm19bT8_19','m198 versus baseline')


#########
#plotting deltas

fig = plt.figure(figsize=(15,10))
fig.suptitle("loss-deltas during training ", fontsize=22)

interval = 5
plt.subplot(2,3,1)
x_caploss, y_caploss =  find_single_caploss (event_19base3, n_64 , interval)  
x_vgsloss, y_vgsloss = find_single_vgsloss (event_19base3, n_64, interval )
delta_y_caploss = ( np.array(y_caploss[1:]) - np.array(y_caploss[:-1]) ) / np.array(y_caploss[:-1])
delta_y_vgsloss = ( np.array(y_vgsloss[1:]) - np.array(y_vgsloss[:-1]) ) / np.array(y_vgsloss[:-1])
plt.plot(x_caploss[1:], np.abs(delta_y_caploss))
plt.plot(x_vgsloss[1:], np.abs(delta_y_vgsloss))
plt.grid()
plt.ylim(0,5)
plt.title("alpha = 0.5")

plt.subplot(2,3,2)
x_caploss, y_caploss =  find_single_caploss (event_19bT3, n_64 , interval)  
x_vgsloss, y_vgsloss = find_single_vgsloss (event_19bT3, n_64, interval )
delta_y_caploss =  ( np.array(y_caploss[1:]) - np.array(y_caploss[:-1]) ) / np.array(y_caploss[:-1])
delta_y_vgsloss =  ( np.array(y_vgsloss[1:]) - np.array(y_vgsloss[:-1]) ) / np.array(y_caploss[:-1])
plt.plot(x_caploss[1:], np.abs(delta_y_caploss))
plt.plot(x_vgsloss[1:], np.abs(delta_y_vgsloss))
plt.grid()
plt.ylim(0,5)
plt.title("alpha = sin")

plt.subplot(2,3,3)
x_caploss, y_caploss =  find_single_caploss (event_19bT5, n_64 , interval)  
x_vgsloss, y_vgsloss = find_single_vgsloss (event_19bT5, n_64, interval )
delta_y_caploss =  ( np.array(y_caploss[1:]) - np.array(y_caploss[:-1]) ) / np.array(y_caploss[:-1])
delta_y_vgsloss =  ( np.array(y_vgsloss[1:]) - np.array(y_vgsloss[:-1]) ) / np.array(y_caploss[:-1])
plt.plot(x_caploss[1:], np.abs(delta_y_caploss))
plt.plot(x_vgsloss[1:], np.abs(delta_y_vgsloss))
plt.grid()
plt.ylim(0,5)
plt.title("alpha = increasing")

plt.subplot(2,3,4)
x_caploss, y_caploss =  find_single_caploss (event_19bT6, n_64 , interval)  
x_vgsloss, y_vgsloss = find_single_vgsloss (event_19bT6, n_64, interval )
delta_y_caploss =  ( np.array(y_caploss[1:]) - np.array(y_caploss[:-1]) ) / np.array(y_caploss[:-1])
delta_y_vgsloss =  ( np.array(y_vgsloss[1:]) - np.array(y_vgsloss[:-1]) ) / np.array(y_caploss[:-1])
plt.plot(x_caploss[1:], np.abs(delta_y_caploss))
plt.plot(x_vgsloss[1:], np.abs(delta_y_vgsloss))
plt.grid()
plt.ylim(0,5)
plt.title("alpha = decreasing")

plt.subplot(2,3,5)
x_caploss, y_caploss =  find_single_caploss (event_19bT7, n_64 , interval)  
x_vgsloss, y_vgsloss = find_single_vgsloss (event_19bT7, n_64, interval )
delta_y_caploss = ( np.array(y_caploss[1:]) - np.array(y_caploss[:-1]) ) / np.array(y_caploss[:-1])
delta_y_vgsloss = ( np.array(y_vgsloss[1:]) - np.array(y_vgsloss[:-1]) ) / np.array(y_vgsloss[:-1])
plt.plot(x_caploss[1:], np.abs(delta_y_caploss))
plt.plot(x_vgsloss[1:], np.abs(delta_y_vgsloss))
plt.grid()
plt.ylim(0,5)
plt.title("alpha = 0.1 ")

plt.subplot(2,3,6)
x_caploss, y_caploss =  find_single_caploss (event_19bT8, n_64 , interval)  
x_vgsloss, y_vgsloss = find_single_vgsloss (event_19bT8, n_64, interval )
delta_y_caploss =  ( np.array(y_caploss[1:]) - np.array(y_caploss[:-1]) ) / np.array(y_caploss[:-1])
delta_y_vgsloss =  ( np.array(y_vgsloss[1:]) - np.array(y_vgsloss[:-1]) ) / np.array(y_caploss[:-1])
plt.plot(x_caploss[1:], np.abs(delta_y_caploss))
plt.plot(x_vgsloss[1:], np.abs(delta_y_vgsloss))
plt.grid()
plt.ylim(0,5)
plt.title("alpha = 0.9 ")

plt.savefig(os.path.join(path_save , 'deltas-vgsloss' + '.png'), format = 'png')

#######################################
def smooth(data):
    kernel_size = 10
    kernel = np.ones(kernel_size) / kernel_size
    data_convolved = np.convolve(data, kernel, mode='same')
    return data_convolved


interval = 100
fig = plt.figure(figsize=(10,10))
fig.suptitle("loss-deltas during training ", fontsize=22)

plt.subplot(2,2,1)
x_caploss, y_caploss =  find_single_caploss (event_19base3, n_64 , interval)  
x_vgsloss, y_vgsloss = find_single_vgsloss (event_19base3, n_64, interval )
delta_y_caploss = ( np.array(y_caploss[1:]) - np.array(y_caploss[:-1]) ) #/ np.array(y_caploss[:-1])
delta_y_vgsloss = ( np.array(y_vgsloss[1:]) - np.array(y_vgsloss[:-1]) ) #/ np.array(y_vgsloss[:-1])
plt.plot(x_caploss[1:], smooth (np.abs(delta_y_caploss)))
#plt.plot(x_vgsloss[1:], smooth (np.abs(delta_y_vgsloss)))
plt.grid()
plt.ylim(0,1)
plt.title("cap-loss simultaneous ")

plt.subplot(2,2,2)
x_caploss, y_caploss =  find_single_caploss (event_19base4, n_64 , interval)  
x_vgsloss, y_vgsloss = find_single_vgsloss (event_19base4, n_64, interval )
delta_y_caploss =  ( np.array(y_caploss[1:]) - np.array(y_caploss[:-1]) ) #/ np.array(y_caploss[:-1])
delta_y_vgsloss =  ( np.array(y_vgsloss[1:]) - np.array(y_vgsloss[:-1]) ) #/ np.array(y_caploss[:-1])
plt.plot(x_caploss[1:], smooth (np.abs(delta_y_caploss)))
#plt.plot(x_vgsloss[1:], smooth (np.abs(delta_y_vgsloss)))
plt.grid()
plt.ylim(0,1)
plt.title("cap-loss pretraining ")

plt.subplot(2,2,3)
x_caploss, y_caploss =  find_single_caploss (event_19base3, n_64 , interval)  
x_vgsloss, y_vgsloss = find_single_vgsloss (event_19base3, n_64, interval )
delta_y_caploss = ( np.array(y_caploss[1:]) - np.array(y_caploss[:-1]) ) #/ np.array(y_caploss[:-1])
delta_y_vgsloss = ( np.array(y_vgsloss[1:]) - np.array(y_vgsloss[:-1]) ) #/ np.array(y_vgsloss[:-1])
plt.plot(x_caploss[1:], smooth (np.abs(delta_y_caploss)))
plt.plot(x_vgsloss[1:], smooth (np.abs(delta_y_vgsloss)))
plt.grid()
plt.ylim(0,5)
plt.title("vgs-loss simultaneous ")

plt.subplot(2,2,4)
x_caploss, y_caploss =  find_single_caploss (event_19base4, n_64 , interval)  
x_vgsloss, y_vgsloss = find_single_vgsloss (event_19base4, n_64, interval )
delta_y_caploss =  ( np.array(y_caploss[1:]) - np.array(y_caploss[:-1]) ) #/ np.array(y_caploss[:-1])
delta_y_vgsloss =  ( np.array(y_vgsloss[1:]) - np.array(y_vgsloss[:-1]) ) #/ np.array(y_caploss[:-1])
plt.plot(x_caploss[1:], smooth (np.abs(delta_y_caploss)))
plt.plot(x_vgsloss[1:], smooth (np.abs(delta_y_vgsloss)))
plt.grid()
plt.ylim(0,5)
plt.title("vgs-loss pretraining ")

plt.savefig(os.path.join(path_save , 'deltas-smooth-base3_4' + '.png'), format = 'png')



