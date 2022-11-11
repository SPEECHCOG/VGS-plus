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
kh
##################################################################
                        ### m6base3  ###
################################################################## 
scores_m6base3 = []
model_name = 'model6base3T'
layer_names = ['L1','L2','L3','L4','L5','L6','L7','L8','L9','L10','L11']
for epoch in range(5,55,10):
    print(epoch)
    for layer_name in layer_names:
        name = 'E' + str(epoch) + layer_name
        print(name) # name = 'E10L3'
        path = os.path.join(path_input, model_name , name , 'score_phonetic.csv')
        name = 'E' + str(epoch) + layer_name
        s = read_score (path)
        scores_m6base3.append(s)

m6base3 = (np.reshape(scores_m6base3, (5,11))).T

##################################################################
                        ### m7base1  ###
################################################################## 
scores_m7base1 = []
model_name = 'model7base1T'
layer_names = ['L1','L2','L3','L4','L5','L6','L7','L8','L9','L10','L11']
for epoch in range(5,45,10):
    print(epoch)
    for layer_name in layer_names:
        name = 'E' + str(epoch) + layer_name
        print(name) # name = 'E10L3'
        path = os.path.join(path_input, model_name , name , 'score_phonetic.csv')
        name = 'E' + str(epoch) + layer_name
        s = read_score (path)
        scores_m7base1.append(s)

m7base1 = (np.reshape(scores_m7base1, (4,11))).T

##################################################################
                        ### testw2v2  ###
################################################################## 
scores_mtest = []
model_name = 'testw2v2'
layer_names = ['L1','L2','L3','L4']
for epoch in [25,30]:
    print(epoch)
    for layer_name in layer_names:
        name = 'E' + str(epoch) + layer_name
        print(name) # name = 'E10L3'
        path = os.path.join(path_input, model_name , name , 'score_phonetic.csv')
        name = 'E' + str(epoch) + layer_name
        s = read_score (path)
        scores_mtest.append(s)

mtest = (np.reshape(scores_mtest, (2,4))).T
##################################################################
                        ### model7base3T  ###
################################################################## 
scores_m7base3 = []
model_name = 'model7base3T'
layer_names = ['L1','L2','L3','L4','L5','L6']
for epoch in [5,15,25]:
    print(epoch)
    for layer_name in layer_names:
        name = 'E' + str(epoch) + layer_name
        print(name) # name = 'E10L3'
        path = os.path.join(path_input, model_name , name , 'score_phonetic.csv')
        name = 'E' + str(epoch) + layer_name
        s = read_score (path)
        scores_m7base1.append(s)

m7base3 = (np.reshape(scores_m7base3, (5,6))).T
################################################################ Plotting

title = 'ABX-error for the original models '
fig = plt.figure(figsize=(15, 10))
fig.suptitle(title, fontsize=20)

plt.subplot(2, 2, 1)  
plt.plot(m6base3[0], label='layer1')
plt.plot(m6base3[1], label='layer2')
plt.plot(m6base3[2], label='layer3')
plt.plot(m6base3[3], label='layer4')
plt.plot(m6base3[4], label='layer5')
plt.plot(m6base3[5], label='layer6')
plt.plot(m6base3[6], label='layer7')
plt.plot(m6base3[7], label='layer8')
plt.plot(m6base3[8], label='layer9')
plt.plot(m6base3[9], label='layer10')
plt.plot(m6base3[10], label='layer11')

plt.title('VGS+ (gradmul = 0.1)',size=14)  
plt.ylabel('abx-error', size=18) 
plt.xticks([0,1,2,3,4],['5', '15', '25', '35', '45'])
plt.grid()
plt.legend(fontsize=14) 

plt.subplot(2, 2, 2)  
plt.plot(m7base1[0], label='layer1')
plt.plot(m7base1[1], label='layer2')
plt.plot(m7base1[2], label='layer3')
plt.plot(m7base1[3], label='layer4')
plt.plot(m7base1[4], label='layer5')
plt.plot(m7base1[5], label='layer6')
plt.plot(m7base1[6], label='layer7')
plt.plot(m7base1[7], label='layer8')
plt.plot(m7base1[8], label='layer9')
plt.plot(m7base1[9], label='layer10')
plt.plot(m7base1[10], label='layer11')
plt.title('w2v2 (gradmul = 1)',size=14)  
plt.ylabel('abx-error', size=18) 
plt.xticks([0,1,2,3],['5', '15', '25', '35'])
plt.grid()
plt.legend(fontsize=14) 

plt.subplot(2, 2, 4)  
plt.plot(mtest[0], label='layer1')
plt.plot(mtest[1], label='layer2')
plt.plot(mtest[2], label='layer3')
plt.plot(mtest[3], label='layer4')
# plt.plot(mtest[4], label='layer5')
# plt.plot(mtest[5], label='layer6')
plt.title('w2v2 (gradmul = 1), batch size = 100',size=14)  
plt.ylabel('abx-error', size=18) 
plt.xticks([0,1],['25', '30'])
plt.grid()
plt.legend(fontsize=14)     

plt.savefig(os.path.join(path_save, 'abx_6_7' + '.png'), format='png')