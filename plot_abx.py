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
                        ### m7ver0  ###
################################################################## 
scores_m7ver0 = []
model_name = 'model7ver0'
layer_names = ['L1','L2','L3','L4','L5','L6','L7','L8']
for epoch in [1,2,3,4,5]:
    print(epoch)
    for layer_name in layer_names:
        name = 'E' + str(epoch) + layer_name
        print(name) # name = 'E10L3'
        path = os.path.join(path_input, model_name , name , 'score_phonetic.csv')
        name = 'E' + str(epoch) + layer_name
        s = read_score (path)
        scores_m7ver0.append(s)

m7ver0 = (np.reshape(scores_m7ver0, (5,8))).T
##################################################################
                        ### m7base1T  ###
################################################################## 
scores_m7base1 = []
model_name = 'model7base1T'
layer_names = ['L1','L2','L3','L4','L5','L6','L7','L8']
for epoch in [5,15,20,25,35]:
    print(epoch)
    for layer_name in layer_names:
        name = 'E' + str(epoch) + layer_name
        print(name) # name = 'E10L3'
        path = os.path.join(path_input, model_name , name , 'score_phonetic.csv')
        name = 'E' + str(epoch) + layer_name
        s = read_score (path)
        scores_m7base1.append(s)

m7base1 = (np.reshape(scores_m7base1, (5,8))).T

##################################################################
                        ### m7base2T  ###
################################################################## 
scores_m7base2 = []
model_name = 'model7base2T'
layer_names = ['L1','L2','L3','L4','L5','L6','L7','L8']
for epoch in [5,15,20,25]:
    print(epoch)
    for layer_name in layer_names:
        name = 'E' + str(epoch) + layer_name
        print(name) # name = 'E10L3'
        path = os.path.join(path_input, model_name , name , 'score_phonetic.csv')
        name = 'E' + str(epoch) + layer_name
        s = read_score (path)
        scores_m7base2.append(s)

m7base2 = (np.reshape(scores_m7base2, (4,8))).T
##################################################################
                        ### model7base3T  ###
################################################################## 
scores_m7base3 = []
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
        scores_m7base3.append(s)


m7base3 = (np.reshape(scores_m7base3, (8,3))).T

##################################################################
                        ### model7ver4  ###
################################################################## 
scores_m7ver4 = []
for i in [16,17,18,19,20,21,22,23]:
    print(scores_m7base1[i])
    scores_m7ver4.append(scores_m7base1[i])
model_name = 'model7ver4'
layer_names = ['L1','L2','L3','L4','L5','L6','L7','L8']
for epoch in [5,15,25, 35, 45]:
    print(epoch)
    for layer_name in layer_names:
        name = 'E' + str(epoch) + layer_name
        print(name) # name = 'E10L3'
        path = os.path.join(path_input, model_name , name , 'score_phonetic.csv')
        name = 'E' + str(epoch) + layer_name
        s = read_score (path)
        scores_m7ver4.append(s)


m7ver4 = (np.reshape(scores_m7ver4, (6,8))).T

##################################################################
                        ### model7ver5  ###
################################################################## 
scores_m7ver5 = []
for i in [16,17,18,19,20,21,22,23]:
    scores_m7ver5.append(scores_m7base2[i])
model_name = 'model7ver5'
layer_names = ['L1','L2','L3','L4','L5','L6','L7','L8']
for epoch in [5,15,25, 35, 45]:
    print(epoch)
    for layer_name in layer_names:
        name = 'E' + str(epoch) + layer_name
        print(name) # name = 'E10L3'
        path = os.path.join(path_input, model_name , name , 'score_phonetic.csv')
        name = 'E' + str(epoch) + layer_name
        s = read_score (path)
        scores_m7ver5.append(s)


m7ver5 = (np.reshape(scores_m7ver5, (6,8))).T
##################################################################
                        ### model7ver6  ###
################################################################## 
scores_m7ver6 = []
for i in [10, 11,12,13,14]:
    scores_m7ver6.append(scores_m7base1[i])
model_name = 'model7ver6'
layer_names = ['L1','L2','L3','L4','L5']#,'L6','L7','L8']
for epoch in [5,15,25, 35, 45]:
    print(epoch)
    for layer_name in layer_names:
        name = 'E' + str(epoch) + layer_name
        print(name) # name = 'E10L3'
        path = os.path.join(path_input, model_name , name , 'score_phonetic.csv')
        name = 'E' + str(epoch) + layer_name
        s = read_score (path)
        scores_m7ver6.append(s)


m7ver6 = (np.reshape(scores_m7ver6, (6,5))).T

##################################################################
                        ### model7ver7  ###
################################################################## 
scores_m7ver7 = []
model_name = 'model7ver7'
layer_names = ['L1','L2','L3','L4','L5','L6','L7','L8']
for epoch in [5, 15, 25, 30]:
    print(epoch)
    for layer_name in layer_names:
        name = 'E' + str(epoch) + layer_name
        print(name) # name = 'E10L3'
        path = os.path.join(path_input, model_name , name , 'score_phonetic.csv')
        name = 'E' + str(epoch) + layer_name
        s = read_score (path)
        scores_m7ver7.append(s)


m7ver7 = (np.reshape(scores_m7ver7, (4,8))).T

##################################################################
                        ### model7ver8  ###
################################################################## 
scores_m7ver8 = []
model_name = 'model7ver8'
layer_names = ['L1','L2','L3','L4','L5','L6','L7','L8']
for epoch in [5, 15, 25, 30, 35, 45, 50]:
    print(epoch)
    for layer_name in layer_names:
        name = 'E' + str(epoch) + layer_name
        print(name) # name = 'E10L3'
        path = os.path.join(path_input, model_name , name , 'score_phonetic.csv')
        name = 'E' + str(epoch) + layer_name
        s = read_score (path)
        scores_m7ver8.append(s)


m7ver8 = (np.reshape(scores_m7ver8, (7,8))).T
################################################################ Plotting

title = 'ABX-error for the original models '
fig = plt.figure(figsize=(15, 20))
fig.suptitle(title, fontsize=20)

plt.subplot(3,3, 1)  
plt.plot(m7ver0[0], label='layer1')
plt.plot(m7ver0[1], label='layer2')
plt.plot(m7ver0[2], label='layer3')
plt.plot(m7ver0[3], label='layer4')
plt.plot(m7ver0[4], label='layer5')
plt.plot(m7ver0[5], label='layer6')
plt.plot(m7ver0[6], label='layer7')
plt.plot(m7ver0[7], label='layer8')
plt.title('base 0: w2v2 pretrained libri ',size=14)  
plt.ylabel('abx-error', size=18) 
plt.xticks([0,1,2,3,4],['1', '2','3', '4', '5'])
plt.grid()
plt.legend(fontsize=14) 


plt.subplot(3,3, 2)  
plt.plot(m7base1[0], label='layer1')
plt.plot(m7base1[1], label='layer2')
plt.plot(m7base1[2], label='layer3')
plt.plot(m7base1[3], label='layer4')
plt.plot(m7base1[4], label='layer5')
# plt.plot(m7base1[5], label='layer6')
# plt.plot(m7base1[6], label='layer7')
# plt.plot(m7base1[7], label='layer8')
# plt.plot(m7base1[8], label='layer9')
# plt.plot(m7base1[9], label='layer10')
# plt.plot(m7base1[10], label='layer11')
plt.title('base 1: w2v2 ',size=14)  
plt.ylabel('abx-error', size=18) 
plt.xticks([0,1,2,3,4,5],['5', '15', '20' ,'25','35','40'])
plt.grid()
plt.legend(fontsize=14)

plt.subplot(3,3, 3)  
plt.plot(m7base2[0], label='layer1')
plt.plot(m7base2[1], label='layer2')
plt.plot(m7base2[2], label='layer3')
plt.plot(m7base2[3], label='layer4')
# plt.plot(m7base2[4], label='layer5')
# plt.plot(m7base2[5], label='layer6')
# plt.plot(m7base2[6], label='layer7')
# plt.plot(m7base2[7], label='layer8')
plt.title('base 2: VGS ',size=14)  
plt.ylabel('abx-error', size=18) 
plt.xticks([0,1,2,3],['5', '15','20', '25'])
plt.grid()
plt.legend(fontsize=14) 


plt.subplot(3,3, 4)  
plt.plot(m7base3[0], label='layer5')
plt.plot(m7base3[1], label='layer6')
plt.plot(m7base3[2], label='layer7')
plt.title('base 3: VGS+',size=14)  
plt.ylabel('abx-error', size=18) 
plt.xticks([0,1,2,3,4,5,6,7],['5', '15', '25' ,'35','45','55','65','75'])
plt.grid()
plt.legend(fontsize=14) 

plt.subplot(3,3, 5)    
plt.plot(m7ver4[0], label='layer1')
plt.plot(m7ver4[1], label='layer2')
plt.plot(m7ver4[2], label='layer3')
plt.plot(m7ver4[3], label='layer4')
plt.plot(m7ver4[4], label='layer5')
# plt.plot(m7ver4[5], label='layer6')
# plt.plot(m7ver4[6], label='layer7')
# plt.plot(m7ver4[7], label='layer8')
plt.title('ver 4: VGS+ pretrained w2v2(E20) ',size=14)  
plt.ylabel('abx-error', size=18) 
#plt.xlabel('epoch', size=18)
plt.xticks([0,1,2,3,4,5],['0','5', '15', '25' ,'35','45'])
plt.grid()
plt.legend(fontsize=14)


plt.subplot(3,3, 6)    
# plt.plot(m7ver5[0], label='layer1')
plt.plot(m7ver5[1], label='layer2')
plt.plot(m7ver5[2], label='layer3')
plt.plot(m7ver5[3], label='layer4')
plt.plot(m7ver5[4], label='layer5')
# plt.plot(m7ver4[5], label='layer6')
# plt.plot(m7ver4[6], label='layer7')
# plt.plot(m7ver4[7], label='layer8')
plt.title('ver 5: VGS+ pretrained VGS(E20) ',size=14)  
plt.ylabel('abx-error', size=18) 
#plt.xlabel('epoch', size=18)
plt.xticks([0,1,2,3,4,5],['0','5', '15', '25' ,'35','45'])
plt.grid()
plt.legend(fontsize=14)


plt.subplot(3,3, 7)    
plt.plot(m7ver6[0], label='layer1')
plt.plot(m7ver6[1], label='layer2')
plt.plot(m7ver6[2], label='layer3')
plt.plot(m7ver6[3], label='layer4')
plt.plot(m7ver6[4], label='layer5')
# plt.plot(m7ver6[5], label='layer6')
# plt.plot(m7ver6[6], label='layer7')
# plt.plot(m7ver6[7], label='layer8')
plt.title('VGS pretrained w2v2(E20) ',size=14)  
plt.ylabel('abx-error', size=18)
#plt.xlabel('epoch', size=18) 
plt.xticks([0,1,2,3,4,5],['0','5', '15', '25' ,'35','45'])
plt.grid()
plt.legend(fontsize=14)


plt.subplot(3,3, 8)    
plt.plot(m7ver7[0], label='layer1')
plt.plot(m7ver7[1], label='layer2')
plt.plot(m7ver7[2], label='layer3')
plt.plot(m7ver7[3], label='layer4')
plt.plot(m7ver7[4], label='layer5')
# plt.plot(m7ver7[5], label='layer6')
# plt.plot(m7ver7[6], label='layer7')
# plt.plot(m7ver7[7], label='layer8')
plt.title('w2v2 pretrained VGS(E20) ',size=14)  
plt.ylabel('abx-error', size=18)
plt.xlabel('epoch', size=18) 
plt.xticks([0,1,2,3],['5', '15', '25' ,'30'])
plt.grid()
plt.legend(fontsize=14) 

plt.subplot(3,3, 9)    
# plt.plot(m7ver8[0], label='layer1')
# plt.plot(m7ver8[1], label='layer2')
# plt.plot(m7ver8[2], label='layer3')
plt.plot(m7ver8[3], label='layer4')
plt.plot(m7ver8[4], label='layer5')
plt.plot(m7ver8[5], label='layer6')
plt.plot(m7ver8[6], label='layer7')
plt.plot(m7ver8[7], label='layer8')
plt.title('w2v2 pretrained VGS+(E20) ',size=14)  
plt.ylabel('abx-error', size=18)
plt.xlabel('epoch', size=18) 
plt.xticks([0,1,2,3,4,5,6],['5', '15', '25' ,'30', '35', '45','50'])
plt.grid()
plt.legend(fontsize=14) 

plt.savefig(os.path.join(path_save, 'abx_base_versions_new' + '.png'), format='png')