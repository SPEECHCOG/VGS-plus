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


def find_single_recall (event, n):
    recall = pd.DataFrame(event.Scalars('acc_r10'))   
    x_recall = [i/n for i in recall['step']]
    y_recall = recall['value'].to_list()    
    return x_recall, y_recall


# Lexical

path_input = "/worktmp2/hxkhkh/current/lextest/output/"

def read_score (path):
    with open(path , 'r') as file:
        a = file.read()
    score = float(a[16:-1])
    return score

############################################################################## 

##################################################################
                        ### m7base1T  ###
################################################################## 
scores_m7base1 = []
model_name = 'model7base1T'
layer_names = ['L1','L2','L3','L4','L5','L6','L7','L8']
for epoch in [5,10,15,25,35,45]:
    print(epoch)
    for layer_name in layer_names:
        name = 'E' + str(epoch) + layer_name
        print(name) # name = 'E10L3'
        path = os.path.join(path_input, 'cls', model_name , name , 'output.txt')
        name = 'E' + str(epoch) + layer_name
        s = read_score (path)
        scores_m7base1.append(s)

m7base1 = (np.reshape(scores_m7base1, (6,8))).T
##################################################################
                        ### model7base3T  ###
################################################################## 
scores_m7base3T = []
model_name = 'model7base3T'
layer_names = ['L1','L2','L3','L4','L5','L6','L7','L8']
for epoch in [1,2,3,4,5,15,25, 35, 45]:
    print(epoch)
    for layer_name in layer_names:
        name = 'E' + str(epoch) + layer_name
        print(name) # name = 'E10L3'
        path = os.path.join(path_input, 'cls', model_name , name , 'output.txt')
        name = 'E' + str(epoch) + layer_name
        s = read_score (path)
        scores_m7base3T.append(s)

m7base3T = (np.reshape(scores_m7base3T, (9,8))).T

##################################################################
                        ### model7ver4T  ###
################################################################## 
scores_m7ver4T = []
model_name = 'model7ver4'
layer_names = ['L1','L2','L3','L4','L5','L6','L7','L8']
for epoch in [1,2,3,4,5,15,25, 35, 45]:
    print(epoch)
    for layer_name in layer_names:
        name = 'E' + str(epoch) + layer_name
        print(name) # name = 'E10L3'
        path = os.path.join(path_input, 'cls', model_name , name , 'output.txt')
        name = 'E' + str(epoch) + layer_name
        s = read_score (path)
        scores_m7ver4T.append(s)


m7ver4T = (np.reshape(scores_m7ver4T, (9,8))).T

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
        path = os.path.join(path_input, 'cls', model_name , name , 'output.txt')
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
        path = os.path.join(path_input, 'cls', model_name , name , 'output.txt')
        name = 'E' + str(epoch) + layer_name
        s = read_score (path)
        scores_m7base5.append(s)

m7base5 = (np.reshape(scores_m7base5, (9,8))).T

# =============================================================================
# ############################################# Normalizing and merging Lexcical
# =============================================================================

x_abx_m7base1 = [0.,5,10,15,25,35,45]
y_abx_m7base1 = []
y_abx_m7base1.extend(np.max(m7base1 , axis = 0)) #best layer performance
y_abx_m7base1.insert(0, 1/89) 

x_abx_m7base3 = [0.,1,2,3,4,5,15,25,35,45]
y_abx_m7base3 = []
y_abx_m7base3.extend(np.max(m7base3T , axis = 0)) #best layer performance
y_abx_m7base3.insert(0, 1/89) 

x_abx_m7ver4 = [0.,1,2,3,4,5,15,25,35,45]
y_abx_m7ver4 = []
y_abx_m7ver4.extend(np.max(m7ver4T , axis = 0)) #best layer performance
bestC_w2v2 = 0.364 
y_abx_m7ver4.insert(0, bestC_w2v2)


x_abx_m7base4 = [0.,1,2,3,4,5,15,25,35,45]
y_abx_m7base4 = []
y_abx_m7base4.extend(np.max(m7base4 , axis = 0)) #best layer performance
y_abx_m7base4.insert(0, 1/89)

x_abx_m7base5 = [0.,1,2,3,4,5,15,25,35,45]
y_abx_m7base5 = []
y_abx_m7base5.extend(np.max(m7base5 , axis = 0)) #best layer performance
y_abx_m7base5.insert(0, 1/89)

max_abx = max(np.max(y_abx_m7base3),np.max(y_abx_m7ver4),np.max(y_abx_m7base4),np.max(y_abx_m7base5))
min_abx = min(np.min(y_abx_m7base3),np.min(y_abx_m7ver4),np.min(y_abx_m7base4),np.min(y_abx_m7base5))

delta_abx = max_abx - min_abx

abx_b1 = [ ((item - min_abx) / delta_abx) for item in y_abx_m7base1]
abx_b3 = [ ((item - min_abx) / delta_abx) for item in y_abx_m7base3]
abx_v4 = [ ((item - min_abx) / delta_abx) for item in y_abx_m7ver4]
abx_b4 = [ ((item - min_abx) / delta_abx) for item in y_abx_m7base4]
abx_b5 = [ ((item - min_abx) / delta_abx) for item in y_abx_m7base5]

# test plot log

x_abx_m7base3 [0] = 0.5
x_abx_m7ver4 [0] = 0.5
x_abx_m7base4 [0] = 0.5
x_abx_m7base5 [0] = 0.5

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(2, 1, 1)
plt.plot(x_abx_m7base3, abx_b3 ,c_1 , label='VGS+ (0.5)')
plt.plot(x_abx_m7ver4, abx_v4 ,c_2 , label='VGS+ Pre')
plt.plot(x_abx_m7base4, abx_b4 ,c_3 , label='VGS+ (0.1)')
plt.plot(x_abx_m7base5, abx_b5 ,c_4 , label='VGS+ (0.9)')
ax.set_xscale('log')
plt.xticks([0.5,1,2,3,4,5,10,50],['0','1','2','3','4','5','10','50'])
plt.ylim(0,1)
plt.ylabel('Normalized lexical score',size=18)
plt.grid()
plt.legend(fontsize=12)
plt.savefig(os.path.join(path_save, 'normalized-lex' + '.png'), format='png')

########################################### saving Recalls as mat file

from scipy.io import savemat

x_abx_m7base3 [0] = 0.
x_abx_m7ver4 [0] = 0.
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


dict_recall['VGSpluspre'] = {}
dict_recall['VGSpluspre']['x'] = x_abx_m7ver4
dict_recall['VGSpluspre']['y'] = y_abx_m7ver4
dict_recall['VGSpluspre']['norm'] = abx_v4

dict_recall['VGSplus01'] = {}
dict_recall['VGSplus01']['x'] = x_abx_m7base4
dict_recall['VGSplus01']['y'] = y_abx_m7base4
dict_recall['VGSplus01']['norm'] = abx_b4

dict_recall['VGSplus09'] = {}
dict_recall['VGSplus09']['x'] = x_abx_m7base5
dict_recall['VGSplus09']['y'] = y_abx_m7base5
dict_recall['VGSplus09']['norm'] = abx_b5

#savemat(path_save + "lex.mat", dict_recall)
