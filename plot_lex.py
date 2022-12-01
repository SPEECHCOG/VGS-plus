import os
import matplotlib.pyplot as plt
import numpy as np

path_input = "/worktmp2/hxkhkh/current/lextest/output/"
path_save = '/worktmp2/hxkhkh/current/FaST/experiments/plots/'
import csv
#path = '/worktmp2/hxkhkh/current/lextest/output/cls/model7base1T/E5L1/output.txt'

        
def read_score (path):
    with open(path , 'r') as file:
        a = file.read()
    score = float(a[16:-1])
    return score


##################################################################
                        ### m7base1T  ###
################################################################## 
scores_m7base1 = []
model_name = 'model7base1T'
layer_names = ['L1','L2','L3','L4','L5','L6','L7','L8']
for epoch in [5,15,25,35,45]:
    print(epoch)
    for layer_name in layer_names:
        name = 'E' + str(epoch) + layer_name
        print(name) # name = 'E10L3'
        path = os.path.join(path_input, 'cls', model_name , name , 'output.txt')
        name = 'E' + str(epoch) + layer_name
        s = read_score (path)
        scores_m7base1.append(s)

m7base1 = (np.reshape(scores_m7base1, (5,8))).T


scoresf_m7base1 = []
model_name = 'model7base1T'
layer_names = ['L1','L2','L3','L4','L5','L6','L7','L8']
for epoch in [5,15,25,35,45]:
    print(epoch)
    for layer_name in layer_names:
        name = 'E' + str(epoch) + layer_name
        print(name) # name = 'E10L3'
        path = os.path.join(path_input, 'frame', model_name , name , 'output.txt')
        name = 'E' + str(epoch) + layer_name
        s = read_score (path)
        scoresf_m7base1.append(s)

m7base1F = (np.reshape(scoresf_m7base1, (5,8))).T
##################################################################
                        ### model7base3T  ###
################################################################## 
scores_m7base3 = []
model_name = 'model7base3T'
layer_names = ['L1','L2','L3','L4','L5','L6','L7','L8']
for epoch in [5,15,25,35,45]:
    print(epoch)
    for layer_name in layer_names:
        name = 'E' + str(epoch) + layer_name
        print(name) # name = 'E10L3'
        path = os.path.join(path_input, 'cls', model_name , name , 'output.txt')
        name = 'E' + str(epoch) + layer_name
        s = read_score (path)
        scores_m7base3.append(s)


m7base3 = (np.reshape(scores_m7base3, (5,8))).T


scoresf_m7base3 = []
model_name = 'model7base3T'
layer_names = ['L1','L2','L3','L4','L5','L6','L7','L8']
for epoch in [5,15,25,35,75]:
    print(epoch)
    for layer_name in layer_names:
        name = 'E' + str(epoch) + layer_name
        print(name) # name = 'E10L3'
        path = os.path.join(path_input, 'frame', model_name , name , 'output.txt')
        name = 'E' + str(epoch) + layer_name
        s = read_score (path)
        scoresf_m7base3.append(s)


m7base3F = (np.reshape(scoresf_m7base3, (5,8))).T
################################################################ Plotting

title = 'lexical performance '
fig = plt.figure(figsize=(15, 10))
fig.suptitle(title, fontsize=20)

plt.subplot(2, 2, 1)  
plt.plot(m7base1[0], label='layer1')
plt.plot(m7base1[1], label='layer2')
plt.plot(m7base1[2], label='layer3')
plt.plot(m7base1[3], label='layer4')
plt.plot(m7base1[4], label='layer5')
plt.plot(m7base1[5], label='layer6')
plt.plot(m7base1[6], label='layer7')
plt.plot(m7base1[7], label='layer8')

plt.title('w2v2, average ',size=14)  
plt.ylabel('abx-error', size=18) 
plt.xticks([0,1,2,3,4],['5', '15', '25','35','45'])
plt.grid()
plt.legend(fontsize=14) 


plt.subplot(2, 2, 2)  
plt.plot(m7base3[0], label='layer1')
plt.plot(m7base3[1], label='layer2')
plt.plot(m7base3[2], label='layer3')
plt.plot(m7base3[3], label='layer4')
plt.plot(m7base3[4], label='layer5')
plt.plot(m7base3[5], label='layer6')
plt.plot(m7base3[6], label='layer7')
plt.plot(m7base3[7], label='layer8')

plt.title('VGS+ average',size=14)  
plt.ylabel('abx-error', size=18) 
plt.xticks([0,1,2,3,4],['5', '15', '25','35','45'])
plt.grid()
plt.legend(fontsize=14) 


plt.subplot(2, 2, 3)  
plt.plot(m7base1F[0], label='layer1')
plt.plot(m7base1F[1], label='layer2')
plt.plot(m7base1F[2], label='layer3')
plt.plot(m7base1F[3], label='layer4')
plt.plot(m7base1F[4], label='layer5')
plt.plot(m7base1F[5], label='layer6')
plt.plot(m7base1F[6], label='layer7')
plt.plot(m7base1F[7], label='layer8')


plt.title('w2v2, frame ',size=14)  
plt.ylabel('abx-error', size=18) 
plt.xticks([0,1,2,3,4],['5', '15', '25','35','45'])
plt.grid()
plt.legend(fontsize=14) 


plt.subplot(2, 2, 4)  
plt.plot(m7base3F[0], label='layer1')
plt.plot(m7base3F[1], label='layer2')
plt.plot(m7base3F[2], label='layer3')
plt.plot(m7base3F[3], label='layer4')
plt.plot(m7base3F[4], label='layer5')
plt.plot(m7base3F[5], label='layer6')
plt.plot(m7base3F[6], label='layer7')
plt.plot(m7base3F[7], label='layer8')

plt.title('VGS+ frame',size=14)  
plt.ylabel('abx-error', size=18) 
plt.xticks([0,1,2,3,4],['5', '15', '25','35','45'])
plt.grid()
plt.legend(fontsize=14)  
plt.savefig(os.path.join(path_save, 'lexical_base13' + '.png'), format='png')