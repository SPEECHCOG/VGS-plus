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

# # Recall
# path_source = '/worktmp2/hxkhkh/current/FaST/experiments/'

# path_event_7base1T = 'model7base1T/'
# path_event_7base3 = 'model7base3/'
# path_event_7base3T = 'model7base3T/'
# path_event_7base3T_extra = 'model7base3T/exp-additional/'

# path_event_7ver4T = 'model7ver4/exp/'
# path_event_7ver4 = 'model7ver4/exp5/'

# path_event_7ver8 = 'model7ver8/exp/'

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

############################################################################## ABX 

##################################################################
                        ### m7base1T  ###
################################################################## 
scores_m7base1 = []
model_name = 'model7base1T'
layer_names = ['L1','L2','L3','L4','L5','L6','L7','L8']
for epoch in [5,10,15,20,25,35,45]:
    print(epoch)
    for layer_name in layer_names:
        name = 'E' + str(epoch) + layer_name
        print(name) # name = 'E10L3'
        path = os.path.join(path_input, model_name , name , 'score_phonetic.csv')
        name = 'E' + str(epoch) + layer_name
        s = read_score (path)
        scores_m7base1.append(s)

m7base1 = (np.reshape(scores_m7base1, (7,8))).T

##################################################################
                        ### model7base3T  ###
################################################################## 
scores_m7base3T = []
model_name = 'model7base3T'
layer_names = ['L1','L2','L3','L4','L5','L6','L7','L8']
for epoch in [1,2,3,4,5,10,15,20,25, 35, 45]:
    print(epoch)
    for layer_name in layer_names:
        name = 'E' + str(epoch) + layer_name
        print(name) # name = 'E10L3'
        path = os.path.join(path_input, model_name , name , 'score_phonetic.csv')
        name = 'E' + str(epoch) + layer_name
        s = read_score (path)
        scores_m7base3T.append(s)

m7base3T = (np.reshape(scores_m7base3T, (11,8))).T


##################################################################
                        ### model7ver4  ###
################################################################## 
scores_m7ver4 = []
layer_names = ['L1','L2','L3','L4','L5','L6','L7','L8']
# Pretrained with w2v2 (20 E)
model_name = 'model7base1T'
epoch = 20
for layer_name in layer_names:
    name = 'E' + str(epoch) + layer_name
    print(name) # name = 'E20L3'
    path = os.path.join(path_input, model_name , name , 'score_phonetic.csv')
    name = 'E' + str(epoch) + layer_name
    s = read_score (path)
    scores_m7ver4.append(s)
    
model_name = 'model7ver4'
for epoch in [1,2,3,4,5,15,25, 35, 45]:
    print(epoch)
    for layer_name in layer_names:
        name = 'E' + str(epoch) + layer_name
        print(name) # name = 'E5L3'
        path = os.path.join(path_input, model_name , name , 'score_phonetic.csv')
        name = 'E' + str(epoch) + layer_name
        s = read_score (path)
        scores_m7ver4.append(s)

m7ver4T = (np.reshape(scores_m7ver4, (10,8))).T

##################################################################
                        ### model7ver8  ###
################################################################## 
scores_m7ver8 = []
layer_names = ['L1','L2','L3','L4','L5','L6','L7','L8']
# Pretrained with VGS+ (20 E)
model_name = 'model7base3T'
epoch = 20
for layer_name in layer_names:
    name = 'E' + str(epoch) + layer_name
    print(name) # name = 'E20L3'
    path = os.path.join(path_input, model_name , name , 'score_phonetic.csv')
    name = 'E' + str(epoch) + layer_name
    s = read_score (path)
    scores_m7ver8.append(s)
    
model_name = 'model7ver8'
for epoch in [1,2,3,4,5,15,25, 35, 45]:
    print(epoch)
    for layer_name in layer_names:
        name = 'E' + str(epoch) + layer_name
        print(name) # name = 'E10L3'
        path = os.path.join(path_input, model_name , name , 'score_phonetic.csv')
        name = 'E' + str(epoch) + layer_name
        s = read_score (path)
        scores_m7ver8.append(s)

m7ver8T = (np.reshape(scores_m7ver8, (10,8))).T

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

# =============================================================================
# ############################################## Normalizing and merging ABX
# =============================================================================

x_abx_m7base1 = [0.0,5,10,15,20,25,35,45]
y_abx_m7base1 = []
y_abx_m7base1.extend(np.min(m7base1 , axis = 0)) #best layer performance
y_abx_m7base1.insert(0, 50)


x_abx_m7base3 = [0.0,1,2,3,4,5,10,15,20,25,35,45]
y_abx_m7base3 = []
y_abx_m7base3.extend(np.min(m7base3T , axis = 0)) #best layer performance
y_abx_m7base3.insert(0, 50)

x_abx_m7ver4 = [0.0,1,2,3,4,5,15,25,35,45]
y_abx_m7ver4 = []
y_abx_m7ver4.extend(np.min(m7ver4T , axis = 0)) #best layer performance

x_abx_m7ver8 = [0.0,1,2,3,4,5,15,25,35,45]
y_abx_m7ver8 = []
y_abx_m7ver8.extend(np.min(m7ver8T , axis = 0)) #best layer performance


x_abx_m7base4 = [0.0,1,2,3,4,5,15,25,35,45]
y_abx_m7base4 = []
y_abx_m7base4.extend(np.min(m7base4 , axis = 0)) #best layer performance
y_abx_m7base4.insert(0, 50)

x_abx_m7base5 = [0.0,1,2,3,4,5,15,25,35,45]
y_abx_m7base5 = []
y_abx_m7base5.extend(np.min(m7base5 , axis = 0)) #best layer performance
y_abx_m7base5.insert(0, 50)


max_abx = max(np.max(y_abx_m7base3),np.max(y_abx_m7ver4),np.max(y_abx_m7base4),np.max(y_abx_m7base5))
min_abx = min(np.min(y_abx_m7base3),np.min(y_abx_m7ver4),np.min(y_abx_m7base4),np.min(y_abx_m7base5))

delta_abx = max_abx - min_abx
abx_b1 = [1- ((item - min_abx) / delta_abx) for item in y_abx_m7base1]
abx_b3 = [1- ((item - min_abx) / delta_abx) for item in y_abx_m7base3]
abx_v4 = [1- ((item - min_abx) / delta_abx) for item in y_abx_m7ver4]
abx_v8 = [1- ((item - min_abx) / delta_abx) for item in y_abx_m7ver8]
abx_b4 = [1- ((item - min_abx) / delta_abx) for item in y_abx_m7base4]
abx_b5 = [1- ((item - min_abx) / delta_abx) for item in y_abx_m7base5]

# test plot log

x_abx_m7base3 [0] = 0.5
x_abx_m7ver4 [0] = 0.5
x_abx_m7ver8 [0] = 0.5
x_abx_m7base4 [0] = 0.5
x_abx_m7base5 [0] = 0.5

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(2, 1, 1)
plt.plot(x_abx_m7base3, abx_b3 ,c_1 , label='VGS+ (0.5)')
plt.plot(x_abx_m7ver4, abx_v4 ,c_2 , label='VGS+ Pre with w2v2')
plt.plot(x_abx_m7ver8, abx_v8 ,c_3 , label='w2v2 Pre with VGS+')
plt.plot(x_abx_m7base4, abx_b4 ,c_4 , label='VGS+ (0.1)')
plt.plot(x_abx_m7base5, abx_b5 ,c_5 , label='VGS+ (0.9)')
ax.set_xscale('log')
plt.xticks([0.5,1,2,3,4,5,10,50],['0','1','2','3','4','5','10','50'])
plt.ylim(0,1)
plt.ylabel('Normalized ABX score',size=18)
plt.grid()
plt.legend(fontsize=12)
plt.savefig(os.path.join(path_save, 'normalized-abx' + '.png'), format='png')


########################################### saving as mat file

from scipy.io import savemat

x_abx_m7base3 [0] = 0.
x_abx_m7ver4 [0] = 0.
x_abx_m7ver8 [0] = 0.
x_abx_m7base4 [0] = 0.
x_abx_m7base5 [0] = 0.


dict_recall = {}

dict_recall['w2v2'] = {}
dict_recall['w2v2']['x'] = x_abx_m7base1
dict_recall['w2v2']['y'] = y_abx_m7base1
dict_recall['w2v2']['norm'] = abx_b1


dict_recall['VGSplus05'] = {}
dict_recall['VGSplus05']['x'] = x_abx_m7base3
dict_recall['VGSplus05']['y'] = y_abx_m7base3
dict_recall['VGSplus05']['norm'] = abx_b3


dict_recall['VGSplusprew2v2'] = {}
dict_recall['VGSplusprew2v2']['x'] = x_abx_m7ver4
dict_recall['VGSplusprew2v2']['y'] = y_abx_m7ver4
dict_recall['VGSplusprew2v2']['norm'] = abx_v4


dict_recall['w2v2preVGSplus'] = {}
dict_recall['w2v2preVGSplus']['x'] = x_abx_m7ver8
dict_recall['w2v2preVGSplus']['y'] = y_abx_m7ver8
dict_recall['w2v2preVGSplus']['norm'] = abx_v8



dict_recall['VGSplus01'] = {}
dict_recall['VGSplus01']['x'] = x_abx_m7base4
dict_recall['VGSplus01']['y'] = y_abx_m7base4
dict_recall['VGSplus01']['norm'] = abx_b4

dict_recall['VGSplus09'] = {}
dict_recall['VGSplus09']['x'] = x_abx_m7base5
dict_recall['VGSplus09']['y'] = y_abx_m7base5
dict_recall['VGSplus09']['norm'] = abx_b5

savemat(path_save + "abx.mat", dict_recall)
