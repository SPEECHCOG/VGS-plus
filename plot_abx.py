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
for epoch in [5,15,25,35,45]:
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
for epoch in [5,15,25]:
    print(epoch)
    for layer_name in layer_names:
        name = 'E' + str(epoch) + layer_name
        print(name) # name = 'E10L3'
        path = os.path.join(path_input, model_name , name , 'score_phonetic.csv')
        name = 'E' + str(epoch) + layer_name
        s = read_score (path)
        scores_m7base2.append(s)

m7base2 = (np.reshape(scores_m7base2, (3,8))).T

##################################################################
                        ### model7base3T  ###
################################################################## 
scores_m7base3 = []
model_name = 'model7base3T'
layer_names = ['L1','L2','L3','L4','L5','L6','L7','L8']
for epoch in [5,15,25,35,45,55,65]:
    print(epoch)
    for layer_name in layer_names:
        name = 'E' + str(epoch) + layer_name
        print(name) # name = 'E10L3'
        path = os.path.join(path_input, model_name , name , 'score_phonetic.csv')
        name = 'E' + str(epoch) + layer_name
        s = read_score (path)
        scores_m7base3.append(s)


m7base3 = (np.reshape(scores_m7base3, (7,8))).T

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
for epoch in [5,15,25, 35, 45]:
    print(epoch)
    for layer_name in layer_names:
        name = 'E' + str(epoch) + layer_name
        print(name) # name = 'E5L3'
        path = os.path.join(path_input, model_name , name , 'score_phonetic.csv')
        name = 'E' + str(epoch) + layer_name
        s = read_score (path)
        scores_m7ver4.append(s)

m7ver4 = (np.reshape(scores_m7ver4, (6,8))).T
##################################################################
                        ### model7ver5  ###
################################################################## 
scores_m7ver5 = []
layer_names = ['L1','L2','L3','L4','L5','L6','L7','L8']
# Pretrained with VGS (20 E)
model_name = 'model7base2T'
epoch = 20
for layer_name in layer_names:
    name = 'E' + str(epoch) + layer_name
    print(name) # name = 'E20L3'
    path = os.path.join(path_input, model_name , name , 'score_phonetic.csv')
    name = 'E' + str(epoch) + layer_name
    s = read_score (path)
    scores_m7ver5.append(s)
   
model_name = 'model7ver5'
for epoch in [5,15,25, 35, 45]:
    print(epoch)
    for layer_name in layer_names:
        name = 'E' + str(epoch) + layer_name
        print(name) # name = 'E5L3'
        path = os.path.join(path_input, model_name , name , 'score_phonetic.csv')
        name = 'E' + str(epoch) + layer_name
        s = read_score (path)
        scores_m7ver5.append(s)
        
m7ver5 = (np.reshape(scores_m7ver5, (6,8))).T
##################################################################
                        ### model7ver6  ###
################################################################## 
scores_m7ver6 = []
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
    scores_m7ver6.append(s)

model_name = 'model7ver6'
for epoch in [5,15,25, 35, 45]:
    print(epoch)
    for layer_name in layer_names:
        name = 'E' + str(epoch) + layer_name
        print(name) # name = 'E10L3'
        path = os.path.join(path_input, model_name , name , 'score_phonetic.csv')
        name = 'E' + str(epoch) + layer_name
        s = read_score (path)
        scores_m7ver6.append(s)


m7ver6 = (np.reshape(scores_m7ver6, (6,8))).T

##################################################################
                        ### model7ver7  ###
##################################################################
scores_m7ver7 = [] 
layer_names = ['L1','L2','L3','L4','L5','L6','L7','L8']
# Pretrained with VGS (20 E)
model_name = 'model7base2T'
epoch = 20
for layer_name in layer_names:
    name = 'E' + str(epoch) + layer_name
    print(name) # name = 'E20L3'
    path = os.path.join(path_input, model_name , name , 'score_phonetic.csv')
    name = 'E' + str(epoch) + layer_name
    s = read_score (path)
    scores_m7ver7.append(s)
    
model_name = 'model7ver7'
for epoch in [5, 15, 25, 35, 45]:
    print(epoch)
    for layer_name in layer_names:
        name = 'E' + str(epoch) + layer_name
        print(name) # name = 'E10L3'
        path = os.path.join(path_input, model_name , name , 'score_phonetic.csv')
        name = 'E' + str(epoch) + layer_name
        s = read_score (path)
        scores_m7ver7.append(s)


m7ver7 = (np.reshape(scores_m7ver7, (6,8))).T
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
for epoch in [5, 15, 25, 35, 45]:
    print(epoch)
    for layer_name in layer_names:
        name = 'E' + str(epoch) + layer_name
        print(name) # name = 'E10L3'
        path = os.path.join(path_input, model_name , name , 'score_phonetic.csv')
        name = 'E' + str(epoch) + layer_name
        s = read_score (path)
        scores_m7ver8.append(s)

m7ver8 = (np.reshape(scores_m7ver8, (6,8))).T
kh
################################################################ Plotting all layers

title = 'ABX-error for best layers '
fig = plt.figure(figsize=(15, 20))
#fig.suptitle(title, fontsize=20)

plt.subplot(3,3, 1)  
x = [1,2,3,4,5]
# plt.plot(m7ver0[0], label='layer1')
# plt.plot(m7ver0[1], label='layer2')
# plt.plot(m7ver0[2], label='layer3')
plt.plot(x, m7ver0[3], label='layer4')
plt.plot(x,m7ver0[4], label='layer5')
# plt.plot(m7ver0[5], label='layer6')
# plt.plot(m7ver0[6], label='layer7')
plt.plot(x,m7ver0[7], label='layer8')
plt.title('base 0: w2v2 pretrained libri ',size=14)  
plt.ylabel('ABX-error', size=18) 
plt.xticks([0,1,2,3,4,5],['0','1', '2','3', '4', '5'])
plt.xlim(-0.2,5.5)
plt.grid()
plt.legend(fontsize=14) 

plt.subplot(3,3, 2) 
x = [5,15,25,35,45] 
plt.plot(x, m7base1[0], label='layer1')
plt.plot(x,m7base1[1], label='layer2')
plt.plot(x,m7base1[2], label='layer3')
# plt.plot(m7base1[3], label='layer4')
# plt.plot(m7base1[4], label='layer5')
# plt.plot(m7base1[5], label='layer6')
# plt.plot(m7base1[6], label='layer7')
# plt.plot(m7base1[7], label='layer8')
plt.title('base 1: w2v2 ',size=14)  
plt.xticks([0,5,15,25,35,45],['0','5', '15', '25','35','45'])
plt.xlim(-2,50.5)
plt.grid()
plt.legend(fontsize=14)

plt.subplot(3,3, 3)  
x = [5,15,25]
# plt.plot(m7base2[0], label='layer1')
plt.plot(x,m7base2[1], label='layer2')
plt.plot(x,m7base2[2], label='layer3')
#plt.plot(m7base2[3], label='layer4')
plt.plot(x,m7base2[4], label='layer5')
# plt.plot(m7base2[5], label='layer6')
# plt.plot(m7base2[6], label='layer7')
# plt.plot(m7base2[7], label='layer8')
plt.title('base 2: VGS ',size=14)  
plt.xticks([0,5,15,25,35,45],['0','5', '15', '25','35','45'])
plt.xlim(-2,50.5)
plt.grid()
plt.legend(fontsize=14) 


plt.subplot(3,3, 4)  
x = [5,15,25,35,45,55,65]
# plt.plot(m7base3[0], label='layer1')
# plt.plot(m7base3[1], label='layer2')
# plt.plot(m7base3[2], label='layer3')
# plt.plot(m7base3[3], label='layer4')
plt.plot(x,m7base3[4], label='layer5')
plt.plot(x,m7base3[5], label='layer6')
plt.plot(x,m7base3[6], label='layer7')
# plt.plot(m7base3[7], label='layer8')
plt.title('base 3: VGS+', size=14)
plt.ylabel('ABX-error', size=18)
plt.xticks([0,5,15,25,35,45,55,65],['0','5', '15', '25','35','45','55','65'])
plt.grid()
plt.legend(fontsize=14)

plt.subplot(3,3, 5)  
x= [0,5,15,25,35,45]  
plt.plot(x,m7ver4[0], label='layer1')
plt.plot(x,m7ver4[1], label='layer2')
plt.plot(x,m7ver4[2], label='layer3')
# plt.plot(m7ver4[3], label='layer4')
# plt.plot(m7ver4[4], label='layer5')
# plt.plot(m7ver4[5], label='layer6')
# plt.plot(m7ver4[6], label='layer7')
# plt.plot(m7ver4[7], label='layer8')
plt.title('ver 4: VGS+ pretrained w2v2(E20) ',size=14)  
plt.xticks([0,5,15,25,35,45],['0','5', '15', '25','35','45'])
plt.xlim(-2,50.5)
plt.grid()
plt.legend(fontsize=14)


plt.subplot(3,3, 6)    
x= [0,5,15,25,35,45] 
# plt.plot(m7ver5[0], label='layer1')
plt.plot(x,m7ver5[1], label='layer2')
# plt.plot(m7ver5[2], label='layer3')
plt.plot(x,m7ver5[3], label='layer4')
plt.plot(x,m7ver5[4], label='layer5')
# plt.plot(m7ver4[5], label='layer6')
# plt.plot(m7ver4[6], label='layer7')
# plt.plot(m7ver4[7], label='layer8')
plt.title('ver 5: VGS+ pretrained VGS(E20) ', size=14)
plt.xticks([0,5,15,25,35,45],['0','5', '15', '25','35','45'])
plt.xlim(-2,50.5)
plt.grid()
plt.legend(fontsize=14)


plt.subplot(3,3, 7)   
x= [0,5,15,25,35,45]  
plt.plot(x,m7ver6[0], label='layer1')
plt.plot(x,m7ver6[1], label='layer2')
plt.plot(x,m7ver6[2], label='layer3')
# plt.plot(m7ver6[3], label='layer4')
# plt.plot(m7ver6[4], label='layer5')
# plt.plot(m7ver6[5], label='layer6')
# plt.plot(m7ver6[6], label='layer7')
# plt.plot(m7ver6[7], label='layer8')
plt.title('ver 6: VGS pretrained w2v2(E20) ',size=14)  
plt.ylabel('ABX-error', size=18)
plt.xlabel('Epoch', size=18) 
plt.xticks([0,5,15,25,35,45],['0','5', '15', '25','35','45'])
plt.xlim(-2,50.5)
plt.grid()
plt.legend(fontsize=14)


plt.subplot(3,3, 8)   
x= [0,5,15,25,35,45]   
plt.plot(x,m7ver7[0], label='layer1')
plt.plot(x,m7ver7[1], label='layer2')
plt.plot(x,m7ver7[2], label='layer3')
# plt.plot(m7ver7[3], label='layer4')
# plt.plot(m7ver7[4], label='layer5')
# plt.plot(m7ver7[5], label='layer6')
# plt.plot(m7ver7[6], label='layer7')
# plt.plot(m7ver7[7], label='layer8')
plt.title('ver 7: w2v2 pretrained VGS(E20) ',size=14)  
plt.xlabel('Epoch', size=18) 
plt.xticks([0,5,15,25,35,45],['0','5', '15', '25','35','45'])
plt.xlim(-2,50.5)
plt.grid()
plt.legend(fontsize=14) 

plt.subplot(3,3, 9)   
x= [0,5,15,25,35,45]   
# plt.plot(m7ver8[0], label='layer1')
# plt.plot(m7ver8[1], label='layer2')
# plt.plot(m7ver8[2], label='layer3')
# plt.plot(m7ver8[3], label='layer4')
# plt.plot(m7ver8[4], label='layer5')
plt.plot(x,m7ver8[5], label='layer6')
plt.plot(x,m7ver8[6], label='layer7')
plt.plot(x,m7ver8[7], label='layer8')
plt.title('ver 8: w2v2 pretrained VGS+(E20) ',size=14)  
plt.ylabel('ABX-error', size=18)
plt.xlabel('Epoch', size=18) 
plt.xticks([0,5,15,25,35,45],['0','5', '15', '25','35','45'])
plt.xlim(-2,50.5)
plt.grid()
plt.legend(fontsize=14) 

plt.savefig(os.path.join(path_save, 'abx_best_layers' + '.png'), format='png')