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


def read_abx (scores, model_name,layer_names, epochs):
    for epoch in epochs:
        print(epoch)
        for layer_name in layer_names:
            name = 'E' + str(epoch) + layer_name
            print(name) # name = 'E10L3'
            path = os.path.join(path_input, model_name , name , 'score_phonetic.csv')
            name = 'E' + str(epoch) + layer_name
            s = read_score (path)
            scores.append(s)
    return scores
    
##################################################################
                        ### m7ver0  ###
################################################################## 
scores = [50,50,50,50,50,50,50,50]
model_name = 'model7ver0'
layer_names = ['L1','L2','L3','L4','L5','L6','L7','L8']
epochs = [1,2,3,4,5]
scores = read_abx (scores,model_name,layer_names, epochs)
ver0 = (np.reshape(scores, (len (epochs)+ 1,len (layer_names)))).T
x_ver0 = epochs
x_ver0.insert(0,0)
z_ver0 = layer_names
##################################################################
                        ### m7base1T  ###
################################################################## 
scores = [50,50,50,50,50,50,50,50]
model_name = 'model7base1T'
layer_names = ['L1','L2','L3','L4','L5','L6','L7','L8']
epochs = [5,15,25,35,45]
scores = read_abx (scores,model_name,layer_names, epochs)
base1 = (np.reshape(scores, (len (epochs) + 1,len (layer_names)))).T
x_base1 = epochs
x_base1.insert(0,0)
z_base1 = layer_names
##################################################################
                        ### m7base2T  ###
################################################################## 
scores = [50,50,50,50,50,50,50,50]
model_name = 'model7base2T'
layer_names = ['L1','L2','L3','L4','L5','L6','L7','L8']
epochs = [5,15,25]
scores = read_abx (scores,model_name,layer_names, epochs)
base2 = (np.reshape(scores, (len (epochs) + 1,len (layer_names)))).T
x_base2 = epochs
x_base2.insert(0,0)
z_base2 = layer_names
##################################################################
                        ### model7base3T  ###
################################################################## 
scores = [50,50,50,50,50,50,50,50]
model_name = 'model7base3T'
layer_names = ['L1','L2','L3','L4','L5','L6','L7','L8']
epochs = [5,15,25,35,45,55,65]
scores = read_abx (scores,model_name,layer_names, epochs)
base3 = (np.reshape(scores, (len (epochs) + 1,len (layer_names)))).T
x_base3 = epochs
x_base3.insert(0,0)
z_base3 = layer_names
##################################################################
                        ### model7ver4  ###
################################################################## 
scores = []
layer_names = ['L1','L2','L3','L4','L5','L6','L7','L8']
# Pretrained with w2v2 (20 E)
model_name = 'model7base1T' 
epochs = [20]
scores = read_abx (scores,model_name,layer_names, epochs)

model_name = 'model7ver4'
epochs = [5,15,25, 35, 45]
scores = read_abx (scores, model_name,layer_names, epochs)

ver4 = (np.reshape(scores, (len (epochs)+1,len (layer_names)))).T
x_ver4 = epochs
x_ver4.insert(0,0)
z_ver4 = layer_names
##################################################################
                        ### model7ver5  ###
################################################################## 
scores = []
layer_names = ['L1','L2','L3','L4','L5','L6','L7','L8']
# Pretrained with VGS (20 E)
model_name = 'model7base2T'
epochs = [20]
scores = read_abx (scores,model_name,layer_names, epochs)
   
model_name = 'model7ver5'
epochs = [5,15,25, 35, 45]
scores = read_abx (scores, model_name,layer_names, epochs)

ver5 = (np.reshape(scores, (len (epochs)+1,len (layer_names)))).T
x_ver5 = epochs
x_ver5.insert(0,0)
z_ver5 = layer_names
##################################################################
                        ### model7ver6  ###
################################################################## 
scores = []
layer_names = ['L1','L2','L3','L4','L5','L6','L7','L8']

# Pretrained with w2v2 (20 E)
model_name = 'model7base1T'
epochs = [20]
scores = read_abx (scores,model_name,layer_names, epochs)

model_name = 'model7ver6'
epochs = [5,15,25, 35, 45]
scores = read_abx (scores, model_name,layer_names, epochs)

ver6 = (np.reshape(scores, (len (epochs)+1,len (layer_names)))).T
x_ver6 = epochs
x_ver6.insert(0,0)
z_ver6 = layer_names
##################################################################
                        ### model7ver7  ###
##################################################################
scores = []
layer_names = ['L1','L2','L3','L4','L5','L6','L7','L8']
# Pretrained with VGS (20 E)
model_name = 'model7base2T'
epochs = [20]
scores = read_abx (scores,model_name,layer_names, epochs)
   
model_name = 'model7ver7'
epochs = [5, 15, 25, 35, 45]
scores = read_abx (scores,model_name,layer_names, epochs)

ver7 = (np.reshape(scores, (len (epochs)+1,len (layer_names)))).T
x_ver7 = epochs
x_ver7.insert(0,0)
z_ver7 = layer_names
##################################################################
                        ### model7ver8  ###
################################################################## 
scores = []
layer_names = ['L1','L2','L3','L4','L5','L6','L7','L8']
# Pretrained with VGS+ (20 E)
model_name = 'model7base3T'
epochs = [20]
scores = read_abx (scores,model_name,layer_names, epochs)


model_name = 'model7ver8'
epochs = [5, 15, 25, 35, 45]
scores = read_abx (scores,model_name,layer_names, epochs)

ver8 = (np.reshape(scores, (len (epochs)+1,len (layer_names)))).T
x_ver8 = epochs
x_ver8.insert(0,0)
z_ver8 = layer_names
################################################################  layers
kh
layers = [1,2,3,4,5,6,7,8]
fig = plt.figure(figsize=(4,8))


ax = fig.add_subplot(2,1,1)
plt.plot(layers, base1[:,-1], label='w2v2')
plt.plot(layers, base2[:,-1], label='VGS')
plt.plot(layers, base3[:,-1], label='VGS+')
plt.plot(layers, ver4[:,-1], label='VGS+ (Pre w2v2)')
plt.plot(layers, ver5[:,-1], label='VGS+ (Pre VGS)')
plt.plot(layers, ver6[:,-1], label='VGS (Pre w2v2)')
plt.plot(layers, ver7[:,-1], label='w2v2 (Pre VGS)')
plt.plot(layers, ver8[:,-1], label='w2v2 (Pre VGS+)')
plt.grid()
plt.legend(fontsize=8) 
plt.ylabel('ABX-error')
plt.xlabel('layer index')
ax.set_yscale('log')
plt.xticks(layers,['1','2','3','4','5','6','7','8'])
plt.yticks([5,6,7,8,9,10,50],['5','6','7','8','9','10','50'])
#plt.savefig(os.path.join(path_save, 'abx_layers-log' + '.png'), format='png')

################################################################  epochs, best score
y_base1 = np.min(base1 , axis = 0) #best layer performance
y_base2 = np.min(base2 , axis = 0)
y_base3 = np.min(base3 , axis = 0)
y_ver4 = np.min(ver4 , axis = 0)
y_ver5 = np.min(ver5 , axis = 0)
y_ver6 = np.min(ver6 , axis = 0)
y_ver7 = np.min(ver7 , axis = 0)
y_ver8 = np.min(ver8 , axis = 0)

#fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(2,1,2)
plt.plot(x_base1, y_base1, label='w2v2')
plt.plot(x_base2, y_base2, label='VGS')
plt.plot(x_base3[0:-2], y_base3[0:-2], label='VGS+')
plt.plot(x_ver4, y_ver4, label='VGS+ (Pre w2v2)')
plt.plot(x_ver5, y_ver5, label='VGS+ (Pre VGS)')
plt.plot(x_ver6, y_ver6, label='VGS (Pre w2v2)')
plt.plot(x_ver7, y_ver7, label='w2v2 (Pre VGS)')
plt.plot(x_ver8, y_ver8, label='w2v2 (Pre VGS+)')
plt.grid()
plt.legend(fontsize=8) 
plt.ylabel('ABX-error')
plt.xlabel('Epoch')
ax.set_yscale('log')
plt.xticks([0,5,15,25,35,45],['0', '5','15','25','35','45'])
plt.yticks([5,6,7,8,9,10,20,50],['5','6','7','8','9','10','20','50'])
#plt.savefig(os.path.join(path_save, 'abxBest_Epochs-log' + '.png'), format='png')
plt.savefig(os.path.join(path_save, 'abx-log' + '.png'), format='png')
################################################################ Plotting all layers all epochs
title = 'ABX-error for best layers '
fig = plt.figure(figsize=(15, 20))
#fig.suptitle(title, fontsize=20)

plt.subplot(3,3, 1)  
x = [1,2,3,4,5]
# plt.plot(m7ver0[0], label='layer1')
# plt.plot(m7ver0[1], label='layer2')
# plt.plot(m7ver0[2], label='layer3')
plt.plot(x, ver0[3], label='layer4')
plt.plot(x,ver0[4], label='layer5')
# plt.plot(m7ver0[5], label='layer6')
# plt.plot(m7ver0[6], label='layer7')
plt.plot(x,ver0[7], label='layer8')
plt.title('base 0: w2v2 pretrained libri ',size=14)  
plt.ylabel('ABX-error', size=18) 
plt.xticks([0,1,2,3,4,5],['0','1', '2','3', '4', '5'])
plt.xlim(-0.2,5.5)
plt.grid()
plt.legend(fontsize=14) 

plt.subplot(3,3, 2) 
x = [5,15,25,35,45] 
plt.plot(x,base1[0], label='layer1')
plt.plot(x,base1[1], label='layer2')
plt.plot(x,base1[2], label='layer3')
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
plt.plot(x,base2[1], label='layer2')
plt.plot(x,base2[2], label='layer3')
#plt.plot(m7base2[3], label='layer4')
plt.plot(x,base2[4], label='layer5')
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
plt.plot(x,base3[4], label='layer5')
plt.plot(x,base3[5], label='layer6')
plt.plot(x,base3[6], label='layer7')
# plt.plot(m7base3[7], label='layer8')
plt.title('base 3: VGS+', size=14)
plt.ylabel('ABX-error', size=18)
plt.xticks([0,5,15,25,35,45,55,65],['0','5', '15', '25','35','45','55','65'])
plt.grid()
plt.legend(fontsize=14)

plt.subplot(3,3, 5)  
x= [0,5,15,25,35,45]  
plt.plot(x,ver4[0], label='layer1')
plt.plot(x,ver4[1], label='layer2')
plt.plot(x,ver4[2], label='layer3')
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
plt.plot(x,ver5[1], label='layer2')
# plt.plot(m7ver5[2], label='layer3')
plt.plot(x,ver5[3], label='layer4')
plt.plot(x,ver5[4], label='layer5')
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
plt.plot(x,ver6[0], label='layer1')
plt.plot(x,ver6[1], label='layer2')
plt.plot(x,ver6[2], label='layer3')
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
plt.plot(x,ver7[0], label='layer1')
plt.plot(x,ver7[1], label='layer2')
plt.plot(x,ver7[2], label='layer3')
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
plt.plot(x,ver8[5], label='layer6')
plt.plot(x,ver8[6], label='layer7')
plt.plot(x,ver8[7], label='layer8')
plt.title('ver 8: w2v2 pretrained VGS+(E20) ',size=14)  
plt.ylabel('ABX-error', size=18)
plt.xlabel('Epoch', size=18) 
plt.xticks([0,5,15,25,35,45],['0','5', '15', '25','35','45'])
plt.xlim(-2,50.5)
plt.grid()
plt.legend(fontsize=14) 

plt.savefig(os.path.join(path_save, 'abx_best_layers' + '.png'), format='png')