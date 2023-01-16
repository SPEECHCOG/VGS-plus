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

path_event_7ver5 = 'model7ver5/exp/'

path_event_7base4T = 'model7base4T/exp/'
path_event_7base5T = 'model7base5T/exp/'


n_32 = 18505
n_64 = 9252

def find_single_recall (event, n):
    recall = pd.DataFrame(event.Scalars('acc_r10'))   
    x_recall = [i/n for i in recall['step']]
    y_recall = recall['value'].to_list()    
    return x_recall, y_recall
kh
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

event_7ver5 =  EventAccumulator(os.path.join(path_source, path_event_7ver5))
event_7ver5.Reload()

event_7base4T =  EventAccumulator(os.path.join(path_source, path_event_7base4T))
event_7base4T.Reload()

event_7base5T =  EventAccumulator(os.path.join(path_source, path_event_7base5T))
event_7base5T.Reload()

############################################################################## Recalls
x_recall_0 = 0
y_recall_0 = 0.002


#........... base1T

x_7base1T_recall, y_7base1T_recall = find_single_recall(event_7base1T, n_64) 
x_7base1T_recall.insert(0, x_recall_0)
y_7base1T_recall.insert(0, y_recall_0)

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

#........... base3

x_7base3_recall, y_7base3_recall = find_single_recall(event_7base3, n_64)
x_7base3_recall.insert(0, x_recall_0)
y_7base3_recall.insert(0, y_recall_0)

#........... ver4T

x_7ver4T_recall, y_7ver4T_recall = find_single_recall(event_7ver4T, n_64) 
x_7ver4T_recall.insert(0, x_recall_0)
y_7ver4T_recall.insert(0, y_recall_0)


#........... ver4

x_7ver4_recall, y_7ver4_recall = find_single_recall(event_7ver4, n_64) 
x_7ver4_recall.insert(0, x_recall_0)
y_7ver4_recall.insert(0, y_recall_0)


#........... ver5T

x_7ver5_recall, y_7ver5_recall = find_single_recall(event_7ver5, n_64) 
x_7ver5_recall.insert(0, x_recall_0)
y_7ver5_recall.insert(0, y_recall_0)


#........... base4T
x_7base4T_recall, y_7base4T_recall = find_single_recall(event_7base4T, n_64) 
x_7base4T_recall.insert(0, x_recall_0)
y_7base4T_recall.insert(0, y_recall_0)

#........... base5T
x_7base5T_recall, y_7base5T_recall = find_single_recall(event_7base5T, n_64) 
x_7base5T_recall.insert(0, x_recall_0)
y_7base5T_recall.insert(0, y_recall_0)

# =============================================================================
# ########################################## merging and normalizing recalls
# =============================================================================

x_recall_m7base3 = []
y_recall_m7base3 = []

x_recall_m7base3.extend(x_7base3_recall[0:5])
y_recall_m7base3.extend(y_7base3_recall[0:5])

x_recall_m7base3.extend(x_7base3T_recall[1:12])
y_recall_m7base3.extend(y_7base3T_recall[1:12])

#.......................................

x_recall_m7ver4 = []
y_recall_m7ver4 = []

x_recall_m7ver4.extend(x_7ver4_recall[0:5])
y_recall_m7ver4.extend(y_7ver4_recall[0:5])

x_recall_m7ver4.extend(x_7ver4T_recall[1:12])
y_recall_m7ver4.extend(y_7ver4T_recall[1:12])

#.......................................

x_recall_m7base4 = []
y_recall_m7base4 = []

x_recall_m7base4.extend(x_7base4T_recall)
y_recall_m7base4.extend(y_7base4T_recall)

#.......................................
x_recall_m7base5 = []
y_recall_m7base5 = []

x_recall_m7base5.extend(x_7base5T_recall)
y_recall_m7base5.extend(y_7base5T_recall)

#.......................................
# normalizing
#.......................................

max_recall = max (np.max(y_recall_m7base3), np.max(y_recall_m7ver4), np.max(y_recall_m7base4), np.max(y_recall_m7base5))
min_recall = min (np.min(y_recall_m7base3), np.min(y_recall_m7ver4), np.min(y_recall_m7base4), np.min(y_recall_m7base5))

delta_recall = max_recall - min_recall

r_b3 = [(item - min_recall) / delta_recall for item in y_recall_m7base3]
r_v4 = [(item - min_recall) / delta_recall for item in y_recall_m7ver4]

r_b4 = [(item - min_recall) / delta_recall for item in y_recall_m7base4]
r_b5 = [(item - min_recall) / delta_recall for item in y_recall_m7base5]

# test plot log

x_recall_m7base3 [0] = 0.5
x_recall_m7ver4 [0] = 0.5
x_recall_m7base4 [0] = 0.5
x_recall_m7base5 [0] = 0.5

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(2, 1, 1)
plt.plot(x_recall_m7base3, r_b3, c_1, label='VGS+ (0.5)')
plt.plot(x_recall_m7ver4, r_v4, c_2, label='VGS+ Pre')
plt.plot(x_recall_m7base4, r_b4, c_3, label='VGS+ (0.1)')
plt.plot(x_recall_m7base5, r_b5, c_4, label='VGS+ (0.9)')
ax.set_xscale('log')
plt.xticks([0.5,1,2,3,4,5,10,50],['0','1','2','3','4','5','10','50'])
plt.ylim(0,1)
plt.ylabel('Normalized recall',size=18)
plt.grid()
plt.legend(fontsize=12)
plt.savefig(os.path.join(path_save, 'normalized-recall' + '.png'), format='png')

########################################### saving Recalls as mat file
kh
from scipy.io import savemat

x_recall_m7base3 [0] = 0
x_recall_m7ver4 [0] = 0
x_recall_m7base4 [0] = 0
x_recall_m7base5 [0] = 0


dict_recall = {}
dict_recall['VGSplus05'] = {}
dict_recall['VGSplus05']['x'] = x_recall_m7base3
dict_recall['VGSplus05']['y'] = y_recall_m7base3
dict_recall['VGSplus05']['norm'] = r_b3

dict_recall['VGSpluspre'] = {}
dict_recall['VGSpluspre']['x'] = x_recall_m7ver4
dict_recall['VGSpluspre']['y'] = y_recall_m7ver4
dict_recall['VGSpluspre']['norm'] = r_v4

dict_recall['VGSplus01'] = {}
dict_recall['VGSplus01']['x'] = x_recall_m7base4
dict_recall['VGSplus01']['y'] = y_recall_m7base4
dict_recall['VGSplus01']['norm'] = r_b4

dict_recall['VGSplus09'] = {}
dict_recall['VGSplus09']['x'] = x_recall_m7base5
dict_recall['VGSplus09']['y'] = y_recall_m7base5
dict_recall['VGSplus09']['norm'] = r_b5

savemat(path_save + "recall-average.mat", dict_recall)
