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


path_event_7base4T = 'model7base4T/exp/'
path_event_7base5T = 'model7base5T/exp/'

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

event_7base4T =  EventAccumulator(os.path.join(path_source, path_event_7base4T))
event_7base4T.Reload()

event_7base5T =  EventAccumulator(os.path.join(path_source, path_event_7base5T))
event_7base5T.Reload()


kh
############################################################################## Recalls
x_recall_0 = 0
y_recall_0 = 0.002

#........... base4T

x_7base4T_recall, y_7base4T_recall = find_single_recall(event_7base4T, n_64) 
x_7base4T_recall.insert(0, x_recall_0)
y_7base4T_recall.insert(0, y_recall_0)

plt.plot(x_7base4T_recall, y_7base4T_recall, c_1, label = '4T')

#........... base5T

x_7base5T_recall, y_7base5T_recall = find_single_recall(event_7base5T, n_64) 
x_7base5T_recall.insert(0, x_recall_0)
y_7base5T_recall.insert(0, y_recall_0)

plt.plot(x_7base5T_recall, y_7base5T_recall, c_1, label = '5T')

############################################################################## ABX 

##################################################################
                        ### m7base4T  ###
################################################################## 
scores_m7base4 = []
model_name = 'model7base4T'
layer_names = ['L1','L2','L3','L4','L5','L6','L7','L8']
for epoch in [1,2,3,4,5,15,25,35,45]:
    print(epoch)
    for layer_name in layer_names:
        name = 'E' + str(epoch) + layer_name
        print(name) # name = 'E10L3'
        path = os.path.join(path_input, model_name , name , 'score_phonetic.csv')
        name = 'E' + str(epoch) + layer_name
        s = read_score (path)
        scores_m7base4.append(s)

m7base4 = (np.reshape(scores_m7base4, (9,8))).T

##################################################################
                        ### m7base5T  ###
################################################################## 
scores_m7base5 = []
model_name = 'model7base5T'
layer_names = ['L1','L2','L3','L4','L5','L6','L7','L8']
for epoch in [1,2,3,4,5,15,25,35,45]:
    print(epoch)
    for layer_name in layer_names:
        name = 'E' + str(epoch) + layer_name
        print(name) # name = 'E10L3'
        path = os.path.join(path_input, model_name , name , 'score_phonetic.csv')
        name = 'E' + str(epoch) + layer_name
        s = read_score (path)
        scores_m7base5.append(s)

m7base5 = (np.reshape(scores_m7base5, (9,8))).T



############################################################################## Plotting ABX (original versions)

title = 'ABX-error'
fig = plt.figure(figsize=(10,5))
fig.suptitle(title, fontsize=20)
plt.subplot(1,2,1)    
#plt.plot(m7base4[0], label='layer1')
plt.plot(m7base4[1], label='layer2')
plt.plot(m7base4[2], label='layer3')
plt.plot(m7base4[3], label='layer4')
plt.plot(m7base4[4], label='layer5')
plt.plot(m7base4[5], label='layer6')
plt.plot(m7base4[6], label='layer7')
plt.plot(m7base4[7], label='layer8')
plt.title('m7base4T  ',size=14)  
plt.ylabel('abx-error', size=18) 
plt.xlabel('epoch', size=18)
plt.xticks([0,1,2,3,4,5,6,7,8],['1','2','3','4','5','15','25','35','45'])
plt.grid()
plt.legend(fontsize=14)

plt.subplot(1,2,2)    
#plt.plot(m7base5[0], label='layer1')
#plt.plot(m7base5[1], label='layer2')
plt.plot(m7base5[2], label='layer3')
plt.plot(m7base5[3], label='layer4')
plt.plot(m7base5[4], label='layer5')
plt.plot(m7base5[5], label='layer6')
plt.plot(m7base5[6], label='layer7')
plt.plot(m7base5[7], label='layer8')
plt.title('m7base5T  ',size=14)  
plt.ylabel('abx-error', size=18) 
plt.xlabel('epoch', size=18)
plt.xticks([0,1,2,3,4,5,6,7,8],['1','2','3','4','5','15','25','35','45'])
plt.grid()
plt.legend(fontsize=14)
plt.savefig(os.path.join(path_save, 'ABX-base4-5' + '.png'), format='png')

############################################################################## lexical
path_input = "/worktmp2/hxkhkh/current/lextest/output/"
##################################################################
                        ### model7base4  ###
##################################################################

scoresF_m7ver4 = []
model_name = 'model7base4T'
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
model_name = 'model7base4T'
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

##################################################################
                        ### model7base5  ###
##################################################################

scoresF_m7ver5 = []
model_name = 'model7base5T'
layer_names = ['L1','L2','L3','L4','L5','L6','L7','L8']
for epoch in [1,2,3,4,5,15,25,35,45]:
    print(epoch)
    for layer_name in layer_names:
        name = 'E' + str(epoch) + layer_name
        print(name) # name = 'E10L3'
        path = os.path.join(path_input, 'frame', model_name , name , 'output.txt')
        name = 'E' + str(epoch) + layer_name
        s = read_lex_score (path)
        scoresF_m7ver5.append(s)

m7Fver5 = (np.reshape(scoresF_m7ver5, (9,8))).T
 
scoresC_m7ver5 = []
model_name = 'model7base5T'
layer_names = ['L1','L2','L3','L4','L5','L6','L7','L8']
for epoch in [1,2,3,4,5,15,25,35,45]:
    print(epoch)
    for layer_name in layer_names:
        name = 'E' + str(epoch) + layer_name
        print(name) # name = 'E10L3'
        path = os.path.join(path_input, 'cls', model_name , name , 'output.txt')
        name = 'E' + str(epoch) + layer_name
        s = read_lex_score (path)
        scoresC_m7ver5.append(s)


m7Cver5 = (np.reshape(scoresC_m7ver5, (9,8))).T

# =============================================================================
# ########################################## Normalizing and merging recalls
# =============================================================================

x_recall_m7ver4 = []
y_recall_m7ver4 = []

x_recall_m7ver4.extend(x_7base4T_recall)
y_recall_m7ver4.extend(y_7base4T_recall)

x_recall_m7ver5 = []
y_recall_m7ver5 = []

x_recall_m7ver5.extend(x_7base5T_recall)
y_recall_m7ver5.extend(y_7base5T_recall)


max_recall_4 = np.max(y_recall_m7ver4)
max_recall_5 = np.max(y_recall_m7ver5)

min_recall_4 = np.min(y_recall_m7ver4)
min_recall_5 = np.min(y_recall_m7ver5)

max_recall = max (max_recall_4, max_recall_5)
min_recall = min (min_recall_4, min_recall_5)

delta_recall = max_recall - min_recall
# delta_recall_3 = max_recall_3 - min_recall_3
# delta_recall_4 = max_recall_4 - min_recall_4

r_4 = [(item - min_recall) / delta_recall for item in y_recall_m7ver4]
r_5 = [(item - min_recall) / delta_recall for item in y_recall_m7ver5]

# =============================================================================
# ############################################## Normalizing and merging ABX
# =============================================================================

x_abx_m7ver4 = [0,1,2,3,4,5,15,25,35,45]
y_abx_m7ver4 = []
y_abx_m7ver4.extend(np.min(m7base4 , axis = 0)) #best layer performance
y_abx_m7ver4.insert(0, 50)

x_abx_m7ver5 = [0,1,2,3,4,5,15,25,35,45]
y_abx_m7ver5 = []
y_abx_m7ver5.extend(np.min(m7base5 , axis = 0)) #best layer performance
y_abx_m7ver5.insert(0, 50)


max_abx_4 = np.max(y_abx_m7ver4)
max_abx_5 = np.max(y_abx_m7ver5)

min_abx_4 = np.min(y_abx_m7ver4)
min_abx_5 = np.min(y_abx_m7ver5)

max_abx = max(max_abx_4,max_abx_5)
min_abx = min(min_abx_4,min_abx_5)

delta_abx = max_abx - min_abx
# delta_abx_3 = max_abx_3 - min_abx_3
# delta_abx_4 = max_abx_4 - min_abx_4

abx_4 = [1- ((item - min_abx) / delta_abx) for item in y_abx_m7ver4]
abx_5 = [1- ((item - min_abx) / delta_abx) for item in y_abx_m7ver5]

# =============================================================================
# ############################################## Normalizing Lexical F
x_lexF_m7ver4 = [0,1,2,3,4,5,15,25,35,45]

y_lexF_m7ver4 = []
y_lexF_m7ver4.extend(np.max(m7Fver4 , axis = 0)) #best layer performance
y_lexF_m7ver4.insert(0, 1/89)

x_lexF_m7ver5 = [0,1,2,3,4,5,15,25,35,45]

y_lexF_m7ver5 = []
y_lexF_m7ver5.extend(np.max(m7Fver5 , axis = 0)) #best layer performance
y_lexF_m7ver5.insert(0, 1/89)

max_lexF_4 = np.max(y_lexF_m7ver4)
max_lexF_5 = np.max(y_lexF_m7ver5)

min_lexF_4= np.min(y_lexF_m7ver4)
min_lexF_5 = np.min(y_lexF_m7ver5)

max_lexF = max(max_lexF_4, max_lexF_5)
min_lexF = min(min_lexF_4, min_lexF_5)

delta_lexF = max_lexF - min_lexF
# delta_lexF_3 = max_lexF_3 - min_lexF_3
# delta_lexF_4 = max_lexF_4 - min_lexF_4

lexF_4 = [(item - min_lexF) / delta_lexF for item in y_lexF_m7ver4]
lexF_5 = [(item - min_lexF) / delta_lexF for item in y_lexF_m7ver5]

# =============================================================================
# ############################################## Normalizing Lexical C : bestC_w2v2 = 0.383
# =============================================================================

x_lexC_m7ver4 = [0,1,2,3,4,5,15,25,35,45]

y_lexC_m7ver4 = []
y_lexC_m7ver4.extend(np.max(m7Cver4 , axis = 0)) #best layer performance
y_lexC_m7ver4.insert(0, 1/89)



x_lexC_m7ver5 = [0,1,2,3,4,5,15,25,35,45]

y_lexC_m7ver5 = []
y_lexC_m7ver5.extend(np.max(m7Cver5 , axis = 0)) #best layer performance
y_lexC_m7ver5.insert(0, 1/89)

max_lexC_4 = np.max(y_lexC_m7ver4)
max_lexC_5 = np.max(y_lexC_m7ver5)

min_lexC_4= np.min(y_lexC_m7ver4)
min_lexC_5 = np.min(y_lexC_m7ver5)

max_lexC = max(max_lexC_4, max_lexC_5)
min_lexC = min(min_lexC_4, min_lexC_5)

delta_lexC = max_lexC - min_lexC
# delta_lexF_3 = max_lexF_3 - min_lexF_3
# delta_lexF_4 = max_lexF_4 - min_lexF_4

lexC_4 = [(item - min_lexC) / delta_lexC for item in y_lexC_m7ver4]
lexC_5 = [(item - min_lexC) / delta_lexC for item in y_lexC_m7ver5]
########################################### Plotting Normalized graphs

fig = plt.figure(figsize=(10,10))

title = 'alpha = 0.1'
#plt.subplot(2, 1, 2)
ax = fig.add_subplot(2, 1, 1)
x_recall_m7ver4 [0] = 0.5
x_abx_m7ver4 [0] = 0.5
x_lexC_m7ver4 [0] = 0.5
x_lexF_m7ver4 [0] = 0.5
plt.plot(x_recall_m7ver4, r_4, c_1, label='recall@10')
plt.plot(x_abx_m7ver4, abx_4,c_3, label='abx')
plt.plot(x_lexC_m7ver4, lexC_4,c_2, label='lex-average')
#plt.plot(x_lexF_m7ver4, lexF_4,c_4, label='lex-frame')
ax.set_xscale('log')
plt.xticks([0.5,1,2,3,4,5,10,50],['0','1','2','3','4','5','10','50'])
plt.ylim(0,1)
plt.ylabel('Normalized performance',size=18)
#plt.xticks([0,1,2,3,4,5,10,15,20,25,30,35,40,45,50],['0','1','2','3','4','5','10','15','20','25','30','35','40','45','50'])
plt.grid()
plt.legend(fontsize=12)
plt.title(title)

title = 'alpha = 0.9 '
#plt.subplot(2, 1, 2)
ax = fig.add_subplot(2, 1, 2)
x_recall_m7ver5 [0] = 0.5
x_abx_m7ver5 [0] = 0.5
x_lexC_m7ver5[0] = 0.5
x_lexF_m7ver5 [0] = 0.5
plt.plot(x_recall_m7ver5, r_5, c_1, label='recall@10')
plt.plot(x_abx_m7ver5, abx_5,c_3, label='abx')
plt.plot(x_lexC_m7ver5, lexC_5,c_2, label='lex-average')
plt.plot(x_lexF_m7ver5, lexF_5,c_4, label='lex-frame')
ax.set_xscale('log')
plt.xticks([0.5,1,2,3,4,5,10,50],['0','1','2','3','4','5','10','50'])
plt.ylim(0,1)
plt.ylabel('Normalized performance',size=18)
#plt.xticks([0,1,2,3,4,5,10,15,20,25,30,35,40,45,50],['0','1','2','3','4','5','10','15','20','25','30','35','40','45','50'])
plt.grid()
plt.legend(fontsize=12)
plt.title(title)

plt.savefig(os.path.join(path_save, 'all_normalized_logx-4-5' + '.png'), format='png')