import os
import matplotlib.pyplot as plt
import numpy as np

path_input = "/worktmp2/hxkhkh/current/ZeroSpeech/output/"
import csv


def read_score (path):
    with open(path , 'r') as file:
      csvreader = csv.reader(file)
      data = []
      for row in csvreader:
        print(row)
        data.append(row)
        
    score = data[1][3]
    return round(100 * float(score) , 2)
##################################################################
                        ### model7ver4  ###
################################################################## 
scores_m7base3 = []
model_name = 'model7base5T'
layer_names = ['L1','L2','L3','L4','L5','L6','L7','L8']
for epoch in [5,15,25,35,45]:
    print(epoch)
    for layer_name in layer_names:
        name = 'E' + str(epoch) + layer_name
        print(name) # name = 'E10L3'
        path = os.path.join(path_input, model_name , name , 'score_phonetic.csv')
        name = 'E' + str(epoch) + layer_name
        s = read_score (path)
        scores_m7base3.append(s)


m7base3 = (np.reshape(scores_m7base3, (5,8))).T


plt.plot(m7base3[0], label='layer1')
plt.plot(m7base3[1], label='layer2')
plt.plot(m7base3[2], label='layer3')
plt.plot(m7base3[3], label='layer4')
plt.plot(m7base3[4], label='layer5')
plt.plot(m7base3[5], label='layer6')
plt.plot(m7base3[6], label='layer7')
plt.plot(m7base3[7], label='layer8')
plt.title('base 3: VGS+', size=14)
plt.ylabel('abx-error', size=18)
plt.xticks([0, 1, 2, 3, 4], ['5', '15', '25', '35', '45'])
plt.grid()
plt.legend(fontsize=14)