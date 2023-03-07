import os
import matplotlib.pyplot as plt
import numpy as np

path_input = "/worktmp2/hxkhkh/current/ZeroSpeech/output/AC/"
path_save = '/worktmp2/hxkhkh/current/FaST/experiments/plots/eusipco/'
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
layer_names = ['L0','L1','L2','L3','L4','L5','L6','L7','L8','L9','L10','L11']
 
c_1 = 'blue'
c_2 = 'green'
c_3 = 'orange'
c_4 = 'red'
c_5 = 'brown'
c_6 = 'darkorange'
c_7 = 'pink'
c_8 = 'royalblue'


label1 = 'W2V2'
label2 = 'VGS'
label3 = 'VGS+'
label4 = '(W2V2, VGS+)'
label5 = '(VGS, VGS+)'
label6 = '(W2V2, VGS)'
label7 = '(VGS, W2V2)'
label8 = '(VGS+, W2V2)'
label9 = '(VGS+, VGS)'
label14 = '(W2V2-35E, VGS+)'
label26 = '(W2V2-5E, VGS)'  
##################################################################
                        ### test  ###
################################################################## 
scores = []
layer_names = ['L0','L1','L2','L3','L4','L5','L6','L7','L8','L9','L10','L11']
model_name = 'model7ver8' 
epochs = [50]
scores = read_abx (scores,model_name,layer_names, epochs)
print('##################################################################')
print('##################################################################')
print(min(scores))
print(np.argmin(scores) + 1)
kh
##################################################################
                        ### m7FB  ###
################################################################## 
scores = []
model_name = 'model7FB'
epochs = [0]
scores = read_abx (scores,model_name,layer_names, epochs)
baseFB = (np.reshape(scores, (len (epochs),len (layer_names)))).T
x_baseFB = epochs
z_baseFB = layer_names

##################################################################
                        ### m7base1T  ###
################################################################## 
scores = [50,50,50,50,50,50,50,50,50,50,50,50]
model_name = 'model7base1T'
epochs = [5,15,25,35,45,55,65,70]
scores = read_abx (scores,model_name,layer_names, epochs)
base1 = (np.reshape(scores, (len (epochs) + 1,len (layer_names)))).T
x_base1 = epochs
x_base1.insert(0,0)
z_base1 = layer_names
##################################################################
                        ### m7base2T  ###
################################################################## 
scores = [50,50,50,50,50,50,50,50,50,50,50,50]
model_name = 'model7base2T'
epochs = [5,15,25,35,45,55,65,70]
scores = read_abx (scores,model_name,layer_names, epochs)
base2 = (np.reshape(scores, (len (epochs) + 1,len (layer_names)))).T
x_base2 = epochs
x_base2.insert(0,0)
z_base2 = layer_names
##################################################################
                        ### model7base3T  ###
################################################################## 
scores = [50,50,50,50,50,50,50,50,50,50,50,50]
model_name = 'model7base3'
epochs = [5,15,25,35,45,55,65,70]
scores = read_abx (scores,model_name,layer_names, epochs)
base3 = (np.reshape(scores, (len (epochs) + 1,len (layer_names)))).T
x_base3 = epochs
x_base3.insert(0,0)
z_base3 = layer_names
##################################################################
                        ### model7ver4  ###
################################################################## 
scores = []
# Pretrained with w2v2 (20 E)
model_name = 'model7base1T' 
epochs = [20]
scores = read_abx (scores,model_name,layer_names, epochs)

model_name = 'model7ver4'
epochs = [5,15,25, 35, 45, 50]
scores = read_abx (scores, model_name,layer_names, epochs)

ver4 = (np.reshape(scores, (len (epochs)+1,len (layer_names)))).T
x_ver4 = epochs
x_ver4.insert(0,0)
z_ver4 = layer_names
##################################################################
                        ### model7ver5  ###
################################################################## 
scores = []
# Pretrained with VGS (20 E)
model_name = 'model7base2T'
epochs = [15]
scores = read_abx (scores,model_name,layer_names, epochs)
   
model_name = 'model7ver5'
epochs = [5,15,25, 35, 45,50]
scores = read_abx (scores, model_name,layer_names, epochs)

ver5 = (np.reshape(scores, (len (epochs)+1,len (layer_names)))).T
x_ver5 = epochs
x_ver5.insert(0,0)
z_ver5 = layer_names
##################################################################
                        ### model7ver6  ###
################################################################## 
scores = []

# Pretrained with w2v2 (20 E)
model_name = 'model7base1T'
epochs = [20]
scores = read_abx (scores,model_name,layer_names, epochs)

model_name = 'model7ver6'
epochs = [5,15,25, 35, 45,50]
scores = read_abx (scores, model_name,layer_names, epochs)

ver6 = (np.reshape(scores, (len (epochs)+1,len (layer_names)))).T
x_ver6 = epochs
x_ver6.insert(0,0)
z_ver6 = layer_names
##################################################################
                        ### model7ver7  ###
##################################################################
scores = []
# Pretrained with VGS (20 E)
model_name = 'model7base2T'
epochs = [15]
scores = read_abx (scores,model_name,layer_names, epochs)
   
model_name = 'model7ver7'
epochs = [5, 15, 25, 35, 45,50]
scores = read_abx (scores,model_name,layer_names, epochs)

ver7 = (np.reshape(scores, (len (epochs)+1,len (layer_names)))).T
x_ver7 = epochs
x_ver7.insert(0,0)
z_ver7 = layer_names
##################################################################
                        ### model7ver8  ###
################################################################## 
scores = []
# Pretrained with VGS+ (20 E)
model_name = 'model7base3'
epochs = [15]
scores = read_abx (scores,model_name,layer_names, epochs)


model_name = 'model7ver8'
epochs = [5, 15, 25, 35, 45,50]
scores = read_abx (scores,model_name,layer_names, epochs)

ver8 = (np.reshape(scores, (len (epochs)+1,len (layer_names)))).T
x_ver8 = epochs
x_ver8.insert(0,0)
z_ver8 = layer_names
##################################################################
                        ### model7ver9  ###
################################################################## 
scores = []
# Pretrained with VGS+ (20 E)
model_name = 'model7base3'
epochs = [15]
scores = read_abx (scores,model_name,layer_names, epochs)


model_name = 'model7ver9'
epochs = [5, 15, 25, 35, 45,50]
scores = read_abx (scores,model_name,layer_names, epochs)

ver9 = (np.reshape(scores, (len (epochs)+1,len (layer_names)))).T
x_ver9 = epochs
x_ver9.insert(0,0)
z_ver9 = layer_names
kh
################################################################  layers

layers = [0,1,2,3,4,5,6,7,8,9,10,11]
fig = plt.figure(figsize=(7,7))
fsize = 14
LW = 4
plt.subplot(1,1,1) #ax = fig.add_subplot(1,1,1)
#plt.plot(layers, baseFB[:,-1], label='FB', lw=LW)
plt.plot(layers, base1[:,-1], c_1, label='W2V2', lw=LW)
plt.plot(layers, base2[:,-1], c_2, label='VGS', lw=LW)
plt.plot(layers, base3[:,-1], c_3, label='VGS+', lw=LW)

plt.plot(layers, ver4[:,-1],c_1, label = label4, linestyle='dashed', lw=LW)# label='(W2V2, VGS+)',
plt.plot(layers, ver5[:,-1],c_2, label = label5, linestyle='dashed' , lw=LW) #label='(VGS, VGS+)'
plt.plot(layers, ver6[:,-1],c_1, label = label6, linestyle='dotted' , lw=LW) #label='(W2V2, VGS)'
plt.plot(layers, ver7[:,-1],c_2, label = label7, linestyle='dotted' , lw=LW) # label='(VGS, W2V2)'
plt.plot(layers, ver8[:,-1],c_3, label = label8, linestyle='dashed' , lw=LW) # label='(VGS+, W2V2)'
plt.grid()
plt.legend(fontsize=fsize) 
plt.ylabel('ABX-error', size=fsize+2)
plt.xlabel('layer index', size=fsize+2)
plt.yscale('log') #ax.set_yscale('log')
plt.xticks(layers,['1','2','3','4','5','6','7','8','9','10','11','12'], size=fsize+2)
plt.yticks([5,6,7,8,9,10,20,30,40,50],['5','6','7','8','9','10','20','30','40','50'], size=fsize+2)
plt.savefig(os.path.join(path_save, 'abx-layers-log-c' + '.pdf'), format='pdf', bbox_inches='tight')
kh
################################################################  epochs, best score

x_ver4 = [i+20 for i in x_ver4]
x_ver5 = [i+20 for i in x_ver5]
x_ver6 = [i+20 for i in x_ver6]
x_ver7 = [i+20 for i in x_ver7]
x_ver8 = [i+20 for i in x_ver8]


y_base1 = np.min(base1 , axis = 0) #best layer performance
y_base2 = np.min(base2 , axis = 0)
y_base3 = np.min(base3 , axis = 0)
y_ver4 = np.min(ver4 , axis = 0)
y_ver5 = np.min(ver5 , axis = 0)
y_ver6 = np.min(ver6 , axis = 0)
y_ver7 = np.min(ver7 , axis = 0)
y_ver8 = np.min(ver8 , axis = 0)

fsize = 14
LW = 4
fig = plt.figure(figsize=(7,7))
plt.subplot(1,1,1) #ax = fig.add_subplot(1,1,1)
plt.plot(x_base1, y_base1,c_1, label='w2v2', lw=LW)
plt.plot(x_base3, y_base3,c_3, label='VGS+', lw=LW)
plt.plot(x_base2, y_base2,c_2, label='VGS', lw=LW)
plt.plot(x_ver4 , y_ver4, c_1, label = label4, linestyle='dashed', lw=LW)
plt.plot(x_ver5, y_ver5, c_2, label = label5, linestyle='dashed', lw=LW)
plt.plot(x_ver6, y_ver6, c_1, label = label6, linestyle='dotted', lw=LW)
plt.plot(x_ver7, y_ver7, c_2, label = label7, linestyle='dotted', lw=LW)
plt.plot(x_ver8, y_ver8, c_3, label = label8, linestyle='dashed', lw=LW)
plt.grid()
plt.legend(fontsize=fsize) 
plt.ylabel('ABX-error', size=fsize+2)
plt.xlabel('Epoch', size=fsize+2)
plt.yscale('log') #ax.set_yscale('log')
plt.xticks([0,5,15,25,35,45,55, 65, 70],['0', '5','15','25','35','45','55','65','70'], size=fsize+2)
plt.yticks([5,6,7,8,9,10,20,30,40,50],['5','6','7','8','9','10','20','30','40','50'], size=fsize+2)
plt.savefig(os.path.join(path_save, 'abx-epochs-log' + '.pdf'), format='pdf', bbox_inches='tight')


################################################################
################################################################ 2 plots (layers, epochs)
################################################################

layers = [0,1,2,3,4,5,6,7,8,9,10,11]
fig = plt.figure(figsize=(14,7))
fsize = 16
LW = 5
plt.subplot(1,2,1) #ax = fig.add_subplot(1,1,1)
#plt.plot(layers, baseFB[:,-1], label='FB', lw=LW)
plt.plot(layers, base1[:,-1], c_1, label='W2V2', lw=LW)
plt.plot(layers, base2[:,-1], c_2, label='VGS', lw=LW)
plt.plot(layers, base3[:,-1], c_3, label='VGS+', lw=LW)

plt.plot(layers, ver4[:,-1],c_1, label = label4, linestyle='dashed', lw=LW)# label='(W2V2, VGS+)',
plt.plot(layers, ver6[:,-1],c_1, label = label6, linestyle='dotted' , lw=LW) #label='(W2V2, VGS)'
plt.plot(layers, ver5[:,-1],c_2, label = label5, linestyle='dashed' , lw=LW) #label='(VGS, VGS+)'
plt.plot(layers, ver7[:,-1],c_2, label = label7, linestyle='dotted' , lw=LW) # label='(VGS, W2V2)'
plt.plot(layers, ver8[:,-1],c_3, label = label8, linestyle='dashed' , lw=LW) # label='(VGS+, W2V2)'
plt.plot(layers, ver9[:,-1],c_3, label = label9, linestyle='dotted' , lw=LW) # label='(VGS+, W2V2)'
plt.grid()
#plt.legend(fontsize=fsize) 
plt.ylim( [4,55] )
plt.ylabel('ABX-error', size=fsize+2)
plt.xlabel('Layer index', size=fsize+2)
plt.yscale('log') #ax.set_yscale('log')
plt.xticks(layers,['1','2','3','4','5','6','7','8','9','10','11','12'], size=fsize+2)
plt.yticks([5,6,7,8,9,10,20,30,40,50],['5','6','7','8','9','10','20','30','40','50'], size=fsize+2)

######################################
x_ver4 = [i+20 for i in x_ver4]
x_ver5 = [i+20 for i in x_ver5]
x_ver6 = [i+20 for i in x_ver6]
x_ver7 = [i+20 for i in x_ver7]
x_ver8 = [i+20 for i in x_ver8]
x_ver9 = [i+20 for i in x_ver9]
y_base1 = np.min(base1 , axis = 0) #best layer performance
y_base2 = np.min(base2 , axis = 0)
y_base3 = np.min(base3 , axis = 0)
y_ver4 = np.min(ver4 , axis = 0)
y_ver5 = np.min(ver5 , axis = 0)
y_ver6 = np.min(ver6 , axis = 0)
y_ver7 = np.min(ver7 , axis = 0)
y_ver8 = np.min(ver8 , axis = 0)
y_ver9 = np.min(ver9 , axis = 0)
plt.subplot(1,2,2) #ax = fig.add_subplot(1,1,1)
plt.plot(x_base1, y_base1,c_1, label='W2V2', lw=LW)
plt.plot(x_base3, y_base3,c_3, label='VGS+', lw=LW)
plt.plot(x_base2, y_base2,c_2, label='VGS', lw=LW)
plt.plot(x_ver4 , y_ver4, c_1, label = label4, linestyle='dashed', lw=LW)
plt.plot(x_ver6, y_ver6, c_1, label = label6, linestyle='dotted', lw=LW)
plt.plot(x_ver5, y_ver5, c_2, label = label5, linestyle='dashed', lw=LW)
plt.plot(x_ver7, y_ver7, c_2, label = label7, linestyle='dotted', lw=LW)
plt.plot(x_ver8, y_ver8, c_3, label = label8, linestyle='dashed', lw=LW)
plt.plot(x_ver9, y_ver9, c_3, label = label9, linestyle='dotted', lw=LW)
plt.grid()
plt.legend(fontsize=fsize )#, bbox_to_anchor=(0.7, 0.4)) # (1.4, 1.2) 
plt.xlabel('Epoch', size=fsize+2)
plt.yscale('log') #ax.set_yscale('log')
plt.ylim( [4,55] )
plt.xticks([0,5,15,25,35,45,55, 65, 70],['0', '5','15','25','35','45','55','65','70'], size=fsize+2)
plt.yticks([5,6,7,8,9,10,20,30,40,50],['5','6','7','8','9','10','20','30','40','50'], size=fsize+2)
plt.savefig(os.path.join(path_save, 'abx-2plots-1row' + '.pdf'), format='pdf', bbox_inches='tight')
################################################################ 
################################################################ Single model
################################################################
layers = [0,1,2,3,4,5,6,7,8,9,10,11]
epochs = [0,5,15,25,35,45,55,65,70]
fig = plt.figure(figsize=(7,7))
fsize = 14
LW = 2
plt.title('VGS',size=fsize+2)
plt.plot(epochs, base2[0,:], label='L0', lw=LW)
# plt.plot(epochs, base2[1,:], label='L1', lw=LW)
#plt.plot(epochs, base2[2,:], label='L2', lw=LW)
plt.plot(epochs, base2[3,:], label='L3', lw=LW)
#plt.plot(epochs, base2[4,:], label='L4', lw=LW)
# plt.plot(epochs, base2[5,:], label='L5', lw=LW)
plt.plot(epochs, base2[6,:], label='L6', lw=LW)
plt.plot(epochs, base2[7,:], label='L7', lw=LW)
plt.plot(epochs, base2[8,:], label='L8',linestyle='dotted', lw=LW)
plt.plot(epochs, base2[9,:], label='L9',linestyle='dotted', lw=LW)
plt.plot(epochs, base2[10,:], label='L10',linestyle='dotted', lw=LW)
plt.plot(epochs, base2[11,:], label='L11', linestyle='dotted', lw=LW)
plt.grid()
plt.legend(fontsize=fsize) 
plt.ylabel('ABX-error', size=fsize+2)
plt.xlabel('Epoch', size=fsize+2)
plt.yscale('log')
plt.ylim( [4.5,10] )
plt.xticks([0,5,15,25,35,45,55, 65, 70],['0', '5','15','25','35','45','55','65','70'], size=fsize+2)
plt.yticks([5,6,7,8,9,10],['5','6','7','8','9','10'], size=fsize+2)
plt.savefig(os.path.join(path_save, 'abx-WC-VGS' + '.pdf'), format='pdf', bbox_inches='tight')
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