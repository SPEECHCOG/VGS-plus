import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv

path_save = '/worktmp2/hxkhkh/current/FaST/experiments/plots/cogsci/'
c_1 = 'blue'
c_2 = 'grey'
c_3 = 'green'
c_4 = 'red'
c_5 = 'brown'
c_6 = 'darkorange'
c_7 = 'pink'
c_8 = 'royalblue'

# Recall
path_source = '/worktmp2/hxkhkh/current/FaST/experiments/'

path_event_7base1T = 'model7base1T/'
path_event_7base3 = 'model7base3/'
path_event_7base3T = 'model7base3T/'
path_event_7base3T_extra = 'model7base3T/exp-additional/'

path_event_7ver4T = 'model7ver4/exp/'
path_event_7ver4 = 'model7ver4/exp5/'

n_32 = 18505
n_64 = 9252

def find_single_recall (event, n):
    recall = pd.DataFrame(event.Scalars('acc_r10'))   
    x_recall = [i/n for i in recall['step']]
    y_recall = recall['value'].to_list()    
    return x_recall, y_recall


# ABX

path_input = "/worktmp2/hxkhkh/current/ZeroSpeech/output/"

def read_score (path):
    with open(path , 'r') as file:
      csvreader = csv.reader(file)
      data = []
      for row in csvreader:
        print(row)
        data.append(row)
        
    score = data[1][3]
    return round(100 * float(score) , 2)

def read_lex_score (path):
    with open(path , 'r') as file:
        a = file.read()
    score = float(a[16:-1])
    return score
############################################################################## Model 7 

event_7base1T =  EventAccumulator(os.path.join(path_source, path_event_7base1T))
event_7base1T.Reload()

event_7base3T =  EventAccumulator(os.path.join(path_source, path_event_7base3T))
event_7base3T.Reload()

event_7base3T_extra =  EventAccumulator(os.path.join(path_source, path_event_7base3T_extra))
event_7base3T_extra.Reload()

event_7base3 =  EventAccumulator(os.path.join(path_source, path_event_7base3))
event_7base3.Reload()

event_7ver4T =  EventAccumulator(os.path.join(path_source, path_event_7ver4T))
event_7ver4T.Reload()

event_7ver4 =  EventAccumulator(os.path.join(path_source, path_event_7ver4))
event_7ver4.Reload()

kh
############################################################################## Recalls
x_recall_0 = 0
y_recall_0 = 0.002

#........... base3T

x_7base3T_recall, y_7base3T_recall = find_single_recall(event_7base3T, n_64)
x_7base3T_recall_extra, y_7base3T_recall_extra = find_single_recall(event_7base3T_extra, n_64)

x_7base3T_recall = x_7base3T_recall[0:-1]
y_7base3T_recall = y_7base3T_recall [0:-1]
for i in range(6):
    x_7base3T_recall.append(55 + (i*5)) 
    y_7base3T_recall.append(y_7base3T_recall_extra[i])

x_7base3T_recall.insert(0, x_recall_0)
y_7base3T_recall.insert(0, y_recall_0)

plt.plot(x_7base3T_recall, y_7base3T_recall, c_1, label = '3T')

#........... base3

x_7base3_recall, y_7base3_recall = find_single_recall(event_7base3, n_64)
x_7base3_recall.insert(0, x_recall_0)
y_7base3_recall.insert(0, y_recall_0)


plt.plot(x_7base3_recall[0:6], y_7base3_recall[0:6], c_1, label = '3')


#........... ver4T

x_7ver4T_recall, y_7ver4T_recall = find_single_recall(event_7ver4T, n_64) 
x_7ver4T_recall.insert(0, x_recall_0)
y_7ver4T_recall.insert(0, y_recall_0)

plt.plot(x_7ver4T_recall, y_7ver4T_recall, c_1, label = '4T')

#........... ver4

x_7ver4_recall, y_7ver4_recall = find_single_recall(event_7ver4, n_64) 
x_7ver4_recall.insert(0, x_recall_0)
y_7ver4_recall.insert(0, y_recall_0)

plt.plot(x_7ver4_recall[0:6], y_7ver4_recall[0:6], c_1, label = '4')

############################################################################## Plotting Recalls (original versions)

fig = plt.figure(figsize=(10,10))

title = 'VGS+ (random init), 5 epochs '
plt.subplot(2,2,1)
plt.plot(x_7base3_recall[0:6], y_7base3_recall[0:6], c_1, label = '3')
plt.ylabel('recall@10',size=18)
plt.ylim(0,1)
plt.grid()
plt.legend(fontsize=12)
plt.title(title)

title = 'VGS+ (random init), 50 epochs '
plt.subplot(2,2,2)
plt.plot(x_7base3T_recall, y_7base3T_recall, c_1, label = '3T')
plt.ylabel('recall@10',size=18)
plt.ylim(0,1)
plt.grid()
plt.legend(fontsize=12)
plt.title(title)

title = 'VGS+ (Pretrained), 5 epochs '
plt.subplot(2,2,3)
plt.plot(x_7ver4_recall[0:6], y_7ver4_recall[0:6], c_1, label = '4')
plt.ylabel('recall@10',size=18)
plt.xlabel('epoch',size=18)
plt.ylim(0,1)
plt.grid()
plt.legend(fontsize=12)
plt.title(title)

title = 'VGS+ (Pretrained), 50 epochs '
plt.subplot(2,2,4)
plt.plot(x_7ver4T_recall, y_7ver4T_recall, c_1, label = '4T')
plt.ylabel('recall@10',size=18)
plt.xlabel('epoch',size=18)
plt.ylim(0,1)
plt.grid()
plt.legend(fontsize=12)
plt.title(title)

plt.savefig(os.path.join(path_save, 'recall-original' + '.png'), format='png')

############################################################################## ABX 

##################################################################
                        ### m7base1T  ###
################################################################## 
scores_m7base1 = []
model_name = 'model7base1T'
layer_names = ['L1','L2','L3','L4','L5']
for epoch in [5,15,20,25,35,40]:
    print(epoch)
    for layer_name in layer_names:
        name = 'E' + str(epoch) + layer_name
        print(name) # name = 'E10L3'
        path = os.path.join(path_input, model_name , name , 'score_phonetic.csv')
        name = 'E' + str(epoch) + layer_name
        s = read_score (path)
        scores_m7base1.append(s)

m7base1 = (np.reshape(scores_m7base1, (6,5))).T
##################################################################
                        ### model7base3T  ###
################################################################## 
scores_m7base3T = []
model_name = 'model7base3T'
layer_names = ['L5','L6','L7']
for epoch in [5,15,25, 35, 45, 55, 65, 75]:
    print(epoch)
    for layer_name in layer_names:
        name = 'E' + str(epoch) + layer_name
        print(name) # name = 'E10L3'
        path = os.path.join(path_input, model_name , name , 'score_phonetic.csv')
        name = 'E' + str(epoch) + layer_name
        s = read_score (path)
        scores_m7base3T.append(s)


m7base3T = (np.reshape(scores_m7base3T, (8,3))).T

##################################################################
                        ### model7base3  ###
################################################################## 
scores_m7base3 = []
model_name = 'model7base3'
layer_names = ['L5','L6','L7']
for epoch in [1,2,3,4,5]:
    print(epoch)
    for layer_name in layer_names:
        name = 'E' + str(epoch) + layer_name
        print(name) # name = 'E10L3'
        path = os.path.join(path_input, model_name , name , 'score_phonetic.csv')
        name = 'E' + str(epoch) + layer_name
        s = read_score (path)
        scores_m7base3.append(s)

m7base3 = (np.reshape(scores_m7base3, (5,3))).T

##################################################################
                        ### model7ver4T  ###
################################################################## 
scores_m7ver4T = []
for i in [10, 11,12,13,14]:
    scores_m7ver4T.append(scores_m7base1[i])
model_name = 'model7ver4'
layer_names = ['L1','L2','L3','L4','L5']#,'L6','L7','L8']
for epoch in [5,15,25, 35, 45]:
    print(epoch)
    for layer_name in layer_names:
        name = 'E' + str(epoch) + layer_name
        print(name) # name = 'E10L3'
        path = os.path.join(path_input, model_name , name , 'score_phonetic.csv')
        name = 'E' + str(epoch) + layer_name
        s = read_score (path)
        scores_m7ver4T.append(s)


m7ver4T = (np.reshape(scores_m7ver4T, (6,5))).T

##################################################################
                        ### model7ver4  ###
################################################################## 
scores_m7ver4 = []
for i in [10, 11,12,13,14]:
    scores_m7ver4.append(scores_m7base1[i])
model_name = 'model7ver4'
layer_names = ['L1','L2','L3','L4','L5']#,'L6','L7','L8']
for epoch in [1,2,3,4,5]:
    print(epoch)
    for layer_name in layer_names:
        name = 'E' + str(epoch) + layer_name
        print(name) # name = 'E10L3'
        path = os.path.join(path_input, model_name , name , 'score_phonetic.csv')
        name = 'E' + str(epoch) + layer_name
        s = read_score (path)
        scores_m7ver4.append(s)

m7ver4 = (np.reshape(scores_m7ver4, (6,5))).T




############################################################################## Plotting ABX (original versions)

title = 'ABX-error'
fig = plt.figure(figsize=(10,10))
fig.suptitle(title, fontsize=20)

plt.subplot(2,2,1)  
plt.plot(m7base3[0], label='layer5')
plt.plot(m7base3[1], label='layer6')
plt.plot(m7base3[2], label='layer7')
plt.title('VGS+ (random init), 5 epochs ',size=14)  
plt.ylabel('abx-error', size=18) 
plt.xticks([1,2,3,4,5],['1','2','3','4','5'])
plt.grid()
plt.legend(fontsize=14) 

plt.subplot(2,2,2)  
plt.plot(m7base3T[0], label='layer5')
plt.plot(m7base3T[1], label='layer6')
plt.plot(m7base3T[2], label='layer7')
plt.title('VGS+ (random init), 50 epochs ',size=14)  
plt.ylabel('abx-error', size=18) 
plt.xticks([1,2,3,4,5,6,7,8],['5', '15', '25' ,'35','45','55','65','75'])
plt.grid()
plt.legend(fontsize=14) 


plt.subplot(2,2,3)    
plt.plot(m7ver4[0], label='layer1')
plt.plot(m7ver4[1], label='layer2')
plt.plot(m7ver4[2], label='layer3')
plt.plot(m7ver4[3], label='layer4')
plt.plot(m7ver4[4], label='layer5')
# plt.plot(m7ver4[5], label='layer6')
# plt.plot(m7ver4[6], label='layer7')
# plt.plot(m7ver4[7], label='layer8')
plt.title('VGS+ (Pretrained), 5 epochs  ',size=14)  
plt.ylabel('abx-error', size=18) 
plt.xlabel('epoch', size=18)
plt.xticks([0,1,2,3,4,5],['0','1','2','3','4','5'])
plt.grid()
plt.legend(fontsize=14)

plt.subplot(2,2,4)    
plt.plot(m7ver4T[0], label='layer1')
plt.plot(m7ver4T[1], label='layer2')
plt.plot(m7ver4T[2], label='layer3')
plt.plot(m7ver4T[3], label='layer4')
plt.plot(m7ver4T[4], label='layer5')
# plt.plot(m7ver4[5], label='layer6')
# plt.plot(m7ver4[6], label='layer7')
# plt.plot(m7ver4[7], label='layer8')
plt.title('VGS+ (Pretrained), 50 epochs ',size=14)  
plt.ylabel('abx-error', size=18) 
plt.xlabel('epoch', size=18)
plt.xticks([0,1,2,3,4,5],['0','5', '15', '25' ,'35','45'])
plt.grid()
plt.legend(fontsize=14)

plt.savefig(os.path.join(path_save, 'ABX-original' + '.png'), format='png')

############################################################################## lexical
path_input = "/worktmp2/hxkhkh/current/lextest/output/"
##################################################################
                        ### m7base1T  ### 
################################################################## 
scoresF_m7base1 = []
model_name = 'model7base1T'
layer_names = ['L1','L2','L3','L4','L5','L6','L7','L8']
for epoch in [5,15,25,35,45]:
    print(epoch)
    for layer_name in layer_names:
        name = 'E' + str(epoch) + layer_name
        print(name) # name = 'E10L3'
        path = os.path.join(path_input, 'cls', model_name , name , 'output.txt')
        name = 'E' + str(epoch) + layer_name
        s = read_lex_score (path)
        scoresF_m7base1.append(s)

m7Fbase1 = (np.reshape(scoresF_m7base1, (5,8))).T


scoresC_m7base1 = []
model_name = 'model7base1T'
layer_names = ['L1','L2','L3','L4','L5','L6','L7','L8']
for epoch in [5,15,25,35,45]:
    print(epoch)
    for layer_name in layer_names:
        name = 'E' + str(epoch) + layer_name
        print(name) # name = 'E10L3'
        path = os.path.join(path_input, 'frame', model_name , name , 'output.txt')
        name = 'E' + str(epoch) + layer_name
        s = read_lex_score (path)
        scoresC_m7base1.append(s)

m7Cbase1 = (np.reshape(scoresC_m7base1, (5,8))).T
##################################################################
                        ### model7base3T  ###
##################################################################

scoresF_m7base3T = []
model_name = 'model7base3T'
layer_names = ['L1','L2','L3','L4','L5','L6','L7','L8']
for epoch in [5,15,25,35,75]:
    print(epoch)
    for layer_name in layer_names:
        name = 'E' + str(epoch) + layer_name
        print(name) # name = 'E10L3'
        path = os.path.join(path_input, 'frame', model_name , name , 'output.txt')
        name = 'E' + str(epoch) + layer_name
        s = read_lex_score (path)
        scoresF_m7base3T.append(s)


m7Fbase3T = (np.reshape(scoresF_m7base3T, (5,8))).T
 
scoresC_m7base3T = []
model_name = 'model7base3T'
layer_names = ['L1','L2','L3','L4','L5','L6','L7','L8']
for epoch in [5,15,25,35,45]:
    print(epoch)
    for layer_name in layer_names:
        name = 'E' + str(epoch) + layer_name
        print(name) # name = 'E10L3'
        path = os.path.join(path_input, 'cls', model_name , name , 'output.txt')
        name = 'E' + str(epoch) + layer_name
        s = read_lex_score (path)
        scoresC_m7base3T.append(s)


m7Cbase3T = (np.reshape(scoresC_m7base3T, (5,8))).T


##################################################################
                        ### model7base3  ###
##################################################################

scoresF_m7base3 = []
model_name = 'model7base3'
layer_names = ['L1','L2','L3','L4','L5','L6','L7','L8']
for epoch in [1,2,3,4,5]:
    print(epoch)
    for layer_name in layer_names:
        name = 'E' + str(epoch) + layer_name
        print(name) # name = 'E10L3'
        path = os.path.join(path_input, 'frame', model_name , name , 'output.txt')
        name = 'E' + str(epoch) + layer_name
        s = read_lex_score (path)
        scoresF_m7base3.append(s)

m7Fbase3 = (np.reshape(scoresF_m7base3, (5,8))).T
 
scoresC_m7base3 = []
model_name = 'model7base3'
layer_names = ['L1','L2','L3','L4','L5','L6','L7','L8']
for epoch in [1,2,3,4,5]:
    print(epoch)
    for layer_name in layer_names:
        name = 'E' + str(epoch) + layer_name
        print(name) # name = 'E10L3'
        path = os.path.join(path_input, 'cls', model_name , name , 'output.txt')
        name = 'E' + str(epoch) + layer_name
        s = read_lex_score (path)
        scoresC_m7base3.append(s)


m7Cbase3 = (np.reshape(scoresC_m7base3, (5,8))).T


##################################################################
                        ### model7ver4  ###
##################################################################

scoresF_m7ver4 = []
model_name = 'model7ver4'
layer_names = ['L1','L2','L3','L4','L5','L6','L7','L8']
for epoch in [1,2,3,4,5,15,25,35,45]:
    print(epoch)
    for layer_name in layer_names:
        name = 'E' + str(epoch) + layer_name
        print(name) # name = 'E10L3'
        path = os.path.join(path_input, 'frame', model_name , name , 'output.txt')
        name = 'E' + str(epoch) + layer_name
        s = read_lex_score (path)
        scoresF_m7ver4.append(s)

m7Fver4 = (np.reshape(scoresF_m7ver4, (9,8))).T
 
scoresC_m7ver4 = []
model_name = 'model7ver4'
layer_names = ['L1','L2','L3','L4','L5','L6','L7','L8']
for epoch in [1,2,3,4,5,15,25,35,45]:
    print(epoch)
    for layer_name in layer_names:
        name = 'E' + str(epoch) + layer_name
        print(name) # name = 'E10L3'
        path = os.path.join(path_input, 'cls', model_name , name , 'output.txt')
        name = 'E' + str(epoch) + layer_name
        s = read_lex_score (path)
        scoresC_m7ver4.append(s)


m7Cver4 = (np.reshape(scoresC_m7ver4, (9,8))).T
################################################################ Plotting

title = 'lexical performance '
fig = plt.figure(figsize=(20,30))
fig.suptitle(title, fontsize=20)

plt.subplot(3, 4, 2)  
plt.plot(m7Cbase1[0], label='layer1')
plt.plot(m7Cbase1[1], label='layer2')
plt.plot(m7Cbase1[2], label='layer3')
plt.plot(m7Cbase1[3], label='layer4')
plt.plot(m7Cbase1[4], label='layer5')
plt.plot(m7Cbase1[5], label='layer6')
plt.plot(m7Cbase1[6], label='layer7')
plt.plot(m7Cbase1[7], label='layer8')
plt.title('w2v2, average ',size=14)  
plt.ylabel('accuracy', size=18) 
plt.xticks([0,1,2,3,4],['5', '15', '25','35','45'])
plt.grid()
plt.legend(fontsize=14) 


plt.subplot(3, 4, 4)  
plt.plot(m7Fbase1[0], label='layer1')
plt.plot(m7Fbase1[1], label='layer2')
plt.plot(m7Fbase1[2], label='layer3')
plt.plot(m7Fbase1[3], label='layer4')
plt.plot(m7Fbase1[4], label='layer5')
plt.plot(m7Fbase1[5], label='layer6')
plt.plot(m7Fbase1[6], label='layer7')
plt.plot(m7Fbase1[7], label='layer8')
plt.title('w2v2, Frame',size=14)  
plt.ylabel('accuracy', size=18) 
plt.xticks([0,1,2,3,4],['5', '15', '25','35','45'])
plt.grid()
plt.legend(fontsize=14) 

#................................................... 3 , 3T
plt.subplot(3, 4, 5)  
plt.plot(m7Cbase3[0], label='layer1')
plt.plot(m7Cbase3[1], label='layer2')
plt.plot(m7Cbase3[2], label='layer3')
plt.plot(m7Cbase3[3], label='layer4')
plt.plot(m7Cbase3[4], label='layer5')
plt.plot(m7Cbase3[5], label='layer6')
plt.plot(m7Cbase3[6], label='layer7')
plt.plot(m7Cbase3[7], label='layer8')
plt.title('VGS+, average ',size=14)  
plt.ylabel('accuracy', size=18) 
plt.xticks([0,1,2,3,4],['1','2','3','4','5'])
plt.grid()
plt.legend(fontsize=14) 


plt.subplot(3, 4, 6)  
plt.plot(m7Cbase3T[0], label='layer1')
plt.plot(m7Cbase3T[1], label='layer2')
plt.plot(m7Cbase3T[2], label='layer3')
plt.plot(m7Cbase3T[3], label='layer4')
plt.plot(m7Cbase3T[4], label='layer5')
plt.plot(m7Cbase3T[5], label='layer6')
plt.plot(m7Cbase3T[6], label='layer7')
plt.plot(m7Cbase3T[7], label='layer8')
plt.title('VGS+, average ',size=14)  
plt.xticks([0,1,2,3,4],['5', '15', '25','35','45'])
plt.grid()
plt.legend(fontsize=14) 


plt.subplot(3, 4, 7)  
plt.plot(m7Fbase3[0], label='layer1')
plt.plot(m7Fbase3[1], label='layer2')
plt.plot(m7Fbase3[2], label='layer3')
plt.plot(m7Fbase3[3], label='layer4')
plt.plot(m7Fbase3[4], label='layer5')
plt.plot(m7Fbase3[5], label='layer6')
plt.plot(m7Fbase3[6], label='layer7')
plt.plot(m7Fbase3[7], label='layer8')
plt.title('VGS+, Frame',size=14)  
plt.xticks([0,1,2,3,4],['1', '2', '3','4','5'])
plt.grid()
plt.legend(fontsize=14) 


plt.subplot(3, 4, 8)  
plt.plot(m7Fbase3T[0], label='layer1')
plt.plot(m7Fbase3T[1], label='layer2')
plt.plot(m7Fbase3T[2], label='layer3')
plt.plot(m7Fbase3T[3], label='layer4')
plt.plot(m7Fbase3T[4], label='layer5')
plt.plot(m7Fbase3T[5], label='layer6')
plt.plot(m7Fbase3T[6], label='layer7')
plt.plot(m7Fbase3T[7], label='layer8')
plt.title('VGS+, Frame',size=14)  
plt.xticks([0,1,2,3,4],['5', '15', '25','35','45'])
plt.grid()
plt.legend(fontsize=14) 


plt.subplot(3, 4, 9)  
plt.plot(m7Cver4[0][0:5], label='layer1')
plt.plot(m7Cver4[1][0:5], label='layer2')
plt.plot(m7Cver4[2][0:5], label='layer3')
plt.plot(m7Cver4[3][0:5], label='layer4')
plt.plot(m7Cver4[4][0:5], label='layer5')
plt.plot(m7Cver4[5][0:5], label='layer6')
plt.plot(m7Cver4[6][0:5], label='layer7')
plt.plot(m7Cver4[7][0:5], label='layer8')
plt.title('VGS+ pretrained, average ',size=14)  
plt.xlabel('epoch', size=18) 
plt.ylabel('accuracy', size=18)
plt.xticks([0,1,2,3,4],['1','2','3','4','5'])
plt.grid()
plt.legend(fontsize=14) 

plt.subplot(3, 4, 10)  
plt.plot(m7Cver4[0][4:], label='layer1')
plt.plot(m7Cver4[1][4:], label='layer2')
plt.plot(m7Cver4[2][4:], label='layer3')
plt.plot(m7Cver4[3][4:], label='layer4')
plt.plot(m7Cver4[4][4:], label='layer5')
plt.plot(m7Cver4[5][4:], label='layer6')
plt.plot(m7Cver4[6][4:], label='layer7')
plt.plot(m7Cver4[7][4:], label='layer8')
plt.title('VGS+ pretrained, average ',size=14)  
plt.xlabel('epoch', size=18)
plt.xticks([0,1,2,3,4],['5', '15', '25','35','45'])
plt.grid()
plt.legend(fontsize=14) 

plt.subplot(3, 4, 11)  
plt.plot(m7Fver4[0][0:5], label='layer1')
plt.plot(m7Fver4[1][0:5], label='layer2')
plt.plot(m7Fver4[2][0:5], label='layer3')
plt.plot(m7Fver4[3][0:5], label='layer4')
plt.plot(m7Fver4[4][0:5], label='layer5')
plt.plot(m7Fver4[5][0:5], label='layer6')
plt.plot(m7Fver4[6][0:5], label='layer7')
plt.plot(m7Fver4[7][0:5], label='layer8')
plt.title('VGS+ pretrained, Frame ',size=14)  
plt.xlabel('epoch', size=18) 
plt.xticks([0,1,2,3,4],['1','2','3','4','5'])
plt.grid()
plt.legend(fontsize=14) 

plt.subplot(3, 4, 12)  
plt.plot(m7Fver4[0][4:], label='layer1')
plt.plot(m7Fver4[1][4:], label='layer2')
plt.plot(m7Fver4[2][4:], label='layer3')
plt.plot(m7Fver4[3][4:], label='layer4')
plt.plot(m7Fver4[4][4:], label='layer5')
plt.plot(m7Fver4[5][4:], label='layer6')
plt.plot(m7Fver4[6][4:], label='layer7')
plt.plot(m7Fver4[7][4:], label='layer8')
plt.title('VGS+ pretrained, Frame ',size=14)  
plt.xlabel('epoch', size=18)
plt.xticks([0,1,2,3,4],['5', '15', '25','35','45'])
plt.grid()
plt.legend(fontsize=14) 

 
plt.savefig(os.path.join(path_save, 'lexical' + '.png'), format='png')


# =============================================================================
# ########################################## Normalizing and merging recalls
# =============================================================================
x_recall_m7base3 = []
y_recall_m7base3 = []

x_recall_m7base3.extend(x_7base3_recall[0:6])
y_recall_m7base3.extend(y_7base3_recall[0:6])

x_recall_m7base3.extend(x_7base3T_recall[2:12])
y_recall_m7base3.extend(y_7base3T_recall[2:12])


x_recall_m7ver4 = []
y_recall_m7ver4 = []

x_recall_m7ver4.extend(x_7ver4_recall[0:6])
y_recall_m7ver4.extend(y_7ver4_recall[0:6])

x_recall_m7ver4.extend(x_7ver4T_recall[2:12])
y_recall_m7ver4.extend(y_7ver4T_recall[2:12])

max_recall_3 = np.max(y_recall_m7base3)
max_recall_4 = np.max(y_7ver4T_recall)

min_recall_3 = np.min(y_recall_m7base3)
min_recall_4 = np.min(y_7ver4T_recall)

max_recall = max (max_recall_3, max_recall_4)
min_recall = min (min_recall_3, min_recall_4)

delta_recall = max_recall - min_recall
# delta_recall_3 = max_recall_3 - min_recall_3
# delta_recall_4 = max_recall_4 - min_recall_4

r_3 = [(item - min_recall) / delta_recall for item in y_recall_m7base3]
r_4 = [(item - min_recall) / delta_recall for item in y_recall_m7ver4]

# =============================================================================
# ############################################## Normalizing and merging ABX
# =============================================================================
# base3: ['1','2','3','4','5']['5', '15', '25' ,'35','45','55','65','75']
# ver4: ['0','1','2','3','4','5']['0','5', '15', '25' ,'35','45']

x_abx_m7base3 = [0,1,2,3,4,5,5,15,25,35,45,55,65,75]
y_abx_m7base3 = []

y_abx_m7base3 = []
y_abx_m7base3.extend(np.min(m7base3 , axis = 0)) #best layer performance
y_abx_m7base3.extend(np.min(m7base3T , axis = 0)) #best layer performance
y_abx_m7base3.insert(0, 50)

x_abx_m7ver4 = [0,1,2,3,4,5,5,15,25,35,45]
y_abx_m7ver4 = []

y_abx_m7ver4 = []
y_abx_m7ver4.extend(np.min(m7ver4 , axis = 0)) #best layer performance
y_abx_m7ver4.extend(np.min(m7ver4T[:,1:] , axis = 0)) #best layer performance

plt.plot(y_abx_m7ver4)

max_abx_3 = np.max(y_abx_m7base3)
max_abx_4 = np.max(y_abx_m7ver4)

min_abx_3 = np.min(y_abx_m7base3)
min_abx_4 = np.min(y_abx_m7ver4)

max_abx = max(max_abx_3,max_abx_4)
min_abx = min(min_abx_3,min_abx_4)

delta_abx = max_abx - min_abx
# delta_abx_3 = max_abx_3 - min_abx_3
# delta_abx_4 = max_abx_4 - min_abx_4

abx_3 = [1- ((item - min_abx) / delta_abx) for item in y_abx_m7base3]
abx_4 = [1- ((item - min_abx) / delta_abx) for item in y_abx_m7ver4]

# =============================================================================
# ############################################## Normalizing Lexical F
# =============================================================================
# base3: ['1','2','3','4','5']['5', '15', '25','35','45']
# ver4: ['1','2','3','4','5']['5', '15', '25','35','45']

x_lexF_m7base3 = [0,1,2,3,4,5,15,25,35,45]
y_lexF_m7base3 = []
y_lexF_m7base3.extend(np.max(m7Fbase3 , axis = 0)) #best layer performance
y_lexF_m7base3.extend(np.max(m7Fbase3T[:,1:] , axis = 0)) #best layer performance
y_lexF_m7base3.insert(0, 1/89)

x_lexF_m7ver4 = [0,1,2,3,4,5,15,25,35,45]
y_lexF_m7ver4 = []

y_lexF_m7ver4 = []
y_lexF_m7ver4.extend(np.max(m7Fver4 , axis = 0)) #best layer performance
bestF_w2v2 = 0.268 
y_lexF_m7ver4.insert(0, bestF_w2v2)

max_lexF_3 = np.max(y_lexF_m7base3)
max_lexF_4 = np.max(y_lexF_m7ver4)

min_lexF_3 = np.min(y_lexF_m7base3)
min_lexF_4 = np.min(y_lexF_m7ver4)

max_lexF = max(max_lexF_3, max_lexF_4)
min_lexF = min(min_lexF_3, min_lexF_4)

delta_lexF = max_lexF - min_lexF
# delta_lexF_3 = max_lexF_3 - min_lexF_3
# delta_lexF_4 = max_lexF_4 - min_lexF_4

lexF_3 = [(item - min_lexF) / delta_lexF for item in y_lexF_m7base3]
lexF_4 = [(item - min_lexF) / delta_lexF for item in y_lexF_m7ver4]

# =============================================================================
# ############################################## Normalizing Lexical C : bestC_w2v2 = 0.383
# =============================================================================

# base3: ['1','2','3','4','5']['5', '15', '25','35','45']
# ver4: ['1','2','3','4','5']['5', '15', '25','35','45']

x_lexC_m7base3 = [0,1,2,3,4,5,15,25,35,45]
y_lexC_m7base3 = []
y_lexC_m7base3.extend(np.max(m7Cbase3 , axis = 0)) #best layer performance
y_lexC_m7base3.extend(np.max(m7Cbase3T[:,1:] , axis = 0)) #best layer performance
y_lexC_m7base3.insert(0, 1/89)

x_lexC_m7ver4 = [0,1,2,3,4,5,15,25,35,45]
y_lexC_m7ver4 = []

y_lexC_m7ver4 = []
y_lexC_m7ver4.extend(np.max(m7Cver4 , axis = 0)) #best layer performance
bestC_w2v2 = 0.383 
y_lexC_m7ver4.insert(0, bestC_w2v2)

max_lexC_3 = np.max(y_lexC_m7base3)
max_lexC_4 = np.max(y_lexC_m7ver4)

min_lexC_3 = np.min(y_lexC_m7base3)
min_lexC_4 = np.min(y_lexC_m7ver4)

max_lexC = max(max_lexC_3,max_lexC_4)
min_lexC = min(min_lexC_3,min_lexC_4)

delta_lexC = max_lexC - min_lexC
# delta_lexC_3 = max_lexC_3 - min_lexC_3
# delta_lexC_4 = max_lexC_4 - min_lexC_4

lexC_3 = [(item - min_lexC) / delta_lexC for item in y_lexC_m7base3]
lexC_4 = [(item - min_lexC) / delta_lexC for item in y_lexC_m7ver4]
########################################### Plotting Normalized graphs

fig = plt.figure(figsize=(10,10))
title = 'VGS+ (random init) '
#plt.subplot(2,1,1)
ax = fig.add_subplot(2, 1, 1)
x_recall_m7base3 [0] = 0.5
x_abx_m7base3 [0] = 0.5
x_lexC_m7base3 [0] = 0.5
x_lexF_m7base3 [0] = 0.5
plt.plot(x_recall_m7base3, r_3, c_1, label = 'recall@10')
plt.plot(x_abx_m7base3[0:-3],abx_3[0:-3],c_3, label='abx')
plt.plot(x_lexC_m7base3, lexC_3,c_2, label='lex-average')
plt.plot(x_lexF_m7base3, lexF_3,c_4, label='lex-frame')
ax.set_xscale('log')
plt.xticks([0.5,1,2,3,4,5,10,50],['0','1','2','3','4','5','10','50'])
plt.ylim(0,1)
plt.ylabel('Normalized performance',size=18)
#plt.xticks([0,1,2,3,4,5,10,15,20,25,30,35,40,45,50],['0','1','2','3','4','5','10','15','20','25','30','35','40','45','50'])
plt.grid()
plt.legend(fontsize=12)
plt.title(title)

title = 'VGS+ (Pretrained) '
#plt.subplot(2, 1, 2)
ax = fig.add_subplot(2, 1, 2)
x_recall_m7ver4 [0] = 0.5
x_abx_m7ver4 [0] = 0.5
x_lexC_m7ver4 [0] = 0.5
x_lexF_m7ver4 [0] = 0.5
plt.plot(x_recall_m7ver4, r_4, c_1, label='recall@10')
plt.plot(x_abx_m7ver4, abx_4,c_3, label='abx')
plt.plot(x_lexC_m7ver4, lexC_4,c_2, label='lex-average')
plt.plot(x_lexF_m7ver4, lexF_4,c_4, label='lex-frame')
ax.set_xscale('log')
plt.xticks([0.5,1,2,3,4,5,10,50],['0','1','2','3','4','5','10','50'])
plt.ylim(0,1)
plt.ylabel('Normalized performance',size=18)
#plt.xticks([0,1,2,3,4,5,10,15,20,25,30,35,40,45,50],['0','1','2','3','4','5','10','15','20','25','30','35','40','45','50'])
plt.grid()
plt.legend(fontsize=12)
plt.title(title)

plt.savefig(os.path.join(path_save, 'all_normalized_logx' + '.png'), format='png')