import os
import matplotlib.pyplot as plt
import numpy as np

path_input = "/worktmp2/hxkhkh/current/ZeroSpeech/output/"
path_save = '/worktmp2/hxkhkh/current/FaST/experiments/plots/'
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


def read_all_scores(path , model_name):
    scores = []

    layer_names = ['L1','L2','L3','L4','L5']
    for epoch in range(5,30,5):
        print(epoch)
        for layer_name in layer_names:
            name = 'E' + str(epoch) + layer_name
            print(name) # name = 'E10L3'
            path = os.path.join(path, model_name,  name , 'score_phonetic.csv')
            name = 'E' + str(epoch) + layer_name
            s = read_score (path)
            scores.append(s)

    scores_out =  ( np.reshape(scores , [5, 5])).T
    return scores_out()
##################################################################
                        ### example  ###
################################################################## 
scores_m6base1 = []
model_name = 'model6base1T'
layer_names = ['L1','L2','L3','L4','L5']
for epoch in range(5,30,5):
    print(epoch)
    for layer_name in layer_names:
        name = 'E' + str(epoch) + layer_name
        print(name) # name = 'E10L3'
        path = os.path.join(path_input, model_name , name , 'score_phonetic.csv')
        name = 'E' + str(epoch) + layer_name
        s = read_score (path)
        scores_m6base1.append(s)

m6base1 =  ( np.reshape(scores_m6base1 , [5, 5])).T

scores_test = read_all_scores(path_input, model_name)
################################################################ Plotting

# title = 'ABX-error for the light models ( grad-mul = 1) '
# fig = plt.figure(figsize=(15, 10))
# fig.suptitle(title, fontsize=20)

# plt.subplot(2, 2, 1)  
# plt.plot(m20base1[0], label='layer1')
# plt.plot(m20base1[1], label='layer2')
# plt.plot(m20base1[2], label='layer3')
# plt.plot(m20base1[3], label='layer4')
# plt.plot(m20base1[4], label='layer5')
# plt.title('w2v2',size=14)  
# plt.ylabel('abx-error', size=18) 
# plt.xticks([0,1,2,3,4],['1', '5', '10', '15', '20'])
# plt.grid()
# plt.legend(fontsize=14)     

