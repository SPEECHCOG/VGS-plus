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
                        ### base 1  ###
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

##################################################################
                        ### base 3  ###
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


##################################################################
                        ### base 4  ###
################################################################## 
scores_m7base4 = []
model_name = 'model7base4T'
layer_names = ['L1','L2','L3','L4','L5','L6','L7','L8']
for epoch in [5,15,25,35,45]:
    print(epoch)
    for layer_name in layer_names:
        name = 'E' + str(epoch) + layer_name
        print(name) # name = 'E10L3'
        path = os.path.join(path_input, 'cls', model_name , name , 'output.txt')
        name = 'E' + str(epoch) + layer_name
        s = read_score (path)
        scores_m7base4.append(s)


m7base4 = (np.reshape(scores_m7base4, (5,8))).T

##################################################################
                        ### base 5  ###
################################################################## 
scores_m7base5 = []
model_name = 'model7base5T'
layer_names = ['L1','L2','L3','L4','L5','L6','L7','L8']
for epoch in [5,15,25,35,45]:
    print(epoch)
    for layer_name in layer_names:
        name = 'E' + str(epoch) + layer_name
        print(name) # name = 'E10L3'
        path = os.path.join(path_input, 'cls', model_name , name , 'output.txt')
        name = 'E' + str(epoch) + layer_name
        s = read_score (path)
        scores_m7base5.append(s)


m7base5 = (np.reshape(scores_m7base5, (5,8))).T

##################################################################
                        ### ver 4  ###
################################################################## 
scores_m7ver4 = []
model_name = 'model7ver4'
layer_names = ['L1','L2','L3','L4','L5','L6','L7','L8']
for epoch in [5,15,25,35,45]:
    print(epoch)
    for layer_name in layer_names:
        name = 'E' + str(epoch) + layer_name
        print(name) # name = 'E10L3'
        path = os.path.join(path_input, 'cls', model_name , name , 'output.txt')
        name = 'E' + str(epoch) + layer_name
        s = read_score (path)
        scores_m7ver4.append(s)


m7ver4 = (np.reshape(scores_m7ver4, (5,8))).T

##################################################################
                        ### ver 5  ###
################################################################## 
scores_m7ver5 = []
model_name = 'model7ver5'
layer_names = ['L1','L2','L3','L4','L5','L6','L7','L8']
for epoch in [5,15,25,35,45]:
    print(epoch)
    for layer_name in layer_names:
        name = 'E' + str(epoch) + layer_name
        print(name) # name = 'E10L3'
        path = os.path.join(path_input, 'cls', model_name , name , 'output.txt')
        name = 'E' + str(epoch) + layer_name
        s = read_score (path)
        scores_m7ver5.append(s)


m7ver5 = (np.reshape(scores_m7ver5, (5,8))).T
################################################################ Plotting

title = 'lexical performance (average over utterance) '
fig = plt.figure(figsize=(15, 10))
fig.suptitle(title, fontsize=20)

plt.subplot(3, 2, 1)  
plt.plot(m7base1[0], label='layer1')
plt.plot(m7base1[1], label='layer2')
plt.plot(m7base1[2], label='layer3')
plt.plot(m7base1[3], label='layer4')
plt.plot(m7base1[4], label='layer5')
plt.plot(m7base1[5], label='layer6')
plt.plot(m7base1[6], label='layer7')
plt.plot(m7base1[7], label='layer8')
plt.title('base 1: w2v2',size=14)  
plt.xticks([0,1,2,3,4],['5', '15', '25','35','45'])
plt.grid()
plt.legend(fontsize=14) 


plt.subplot(3, 2, 2)  
plt.plot(m7base3[0], label='layer1')
plt.plot(m7base3[1], label='layer2')
plt.plot(m7base3[2], label='layer3')
plt.plot(m7base3[3], label='layer4')
plt.plot(m7base3[4], label='layer5')
plt.plot(m7base3[5], label='layer6')
plt.plot(m7base3[6], label='layer7')
plt.plot(m7base3[7], label='layer8')
plt.title('base 3: VGS+',size=14)  
plt.xticks([0,1,2,3,4],['5', '15', '25','35','45'])
plt.grid()
plt.legend(fontsize=14) 

plt.subplot(3, 2, 3)  
plt.plot(m7base4[0], label='layer1')
plt.plot(m7base4[1], label='layer2')
plt.plot(m7base4[2], label='layer3')
plt.plot(m7base4[3], label='layer4')
plt.plot(m7base4[4], label='layer5')
plt.plot(m7base4[5], label='layer6')
plt.plot(m7base4[6], label='layer7')
plt.plot(m7base4[7], label='layer8')
plt.title('base 4: VGS+ (alpha = 0.1)',size=14)  

plt.xticks([0,1,2,3,4],['5', '15', '25','35','45'])
plt.grid()
plt.legend(fontsize=14) 

plt.subplot(3, 2, 4)  
plt.plot(m7base5[0], label='layer1')
plt.plot(m7base5[1], label='layer2')
plt.plot(m7base5[2], label='layer3')
plt.plot(m7base5[3], label='layer4')
plt.plot(m7base5[4], label='layer5')
plt.plot(m7base5[5], label='layer6')
plt.plot(m7base5[6], label='layer7')
plt.plot(m7base5[7], label='layer8')
plt.title('base 5: VGS+ (alpha = 0.9)',size=14)  

plt.xticks([0,1,2,3,4],['5', '15', '25','35','45'])
plt.grid()
plt.legend(fontsize=14)


plt.subplot(3, 2, 5)  
plt.plot(m7ver4[0], label='layer1')
plt.plot(m7ver4[1], label='layer2')
plt.plot(m7ver4[2], label='layer3')
plt.plot(m7ver4[3], label='layer4')
plt.plot(m7ver4[4], label='layer5')
plt.plot(m7ver4[5], label='layer6')
plt.plot(m7ver4[6], label='layer7')
plt.plot(m7ver4[7], label='layer8')
plt.title('ver 4: VGS+ (pretrained w2v2)',size=14)  
 
plt.xticks([0,1,2,3,4],['5', '15', '25','35','45'])
plt.grid()
plt.legend(fontsize=14) 

plt.subplot(3, 2, 6)  
plt.plot(m7ver5[0], label='layer1')
plt.plot(m7ver5[1], label='layer2')
plt.plot(m7ver5[2], label='layer3')
plt.plot(m7ver5[3], label='layer4')
plt.plot(m7ver5[4], label='layer5')
plt.plot(m7ver5[5], label='layer6')
plt.plot(m7ver5[6], label='layer7')
plt.plot(m7ver5[7], label='layer8')
plt.title('ver 5: VGS+ (pretrained VGS)',size=14)  

plt.xticks([0,1,2,3,4],['5', '15', '25','35','45'])
plt.grid()
plt.legend(fontsize=14) 

 
plt.savefig(os.path.join(path_save, 'lexical_cls_versions' + '.png'), format='png')