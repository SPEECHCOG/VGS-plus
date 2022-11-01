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

################################################################# plotting abx for m19base
# m19base1 =  np.array([[0.1816,0.1183,0.1061,0.1024,0.1022],[0.1902,0.1264 ,0.1175,0.1145,0.1167],
#               [0.2025,0.1532,0.1567,0.1604,0.1659],[0.2129,0.1948,0.2233,0.2429,0.2484],
#               [0.2072,0.1950,0.2189,0.2491,0.2546]])

# m19base2 = np.array([[],[],
#               [],[],
#               []])

# m19base3 = np.array([[0.1409,0.066,0.0643, 0.0634, 0.0616],[0.1383, 0.0626, 0.0588, 0.0587,0.0560],
#               [0.1361,0.0594,0.0562,0.0554,0.0524],[0.1396,0.0575,0.0537,0.0534,0.0501],
#               [0.1467,0.0603,0.0547,0.0537,0.0503]])


# m19base4 = np.array([[],[],
#               [],[],
#               []])

# title = 'ABX-error for the light models at different epochs during training'
# fig = plt.figure(figsize=(15, 10))
# fig.suptitle(title, fontsize=20)
# plt.subplot(2, 2, 1)
# plt.plot(m19base1.T[:, 0], label='layer1')
# plt.plot(m19base1.T[:, 1], label='layer2')
# plt.plot(m19base1.T[:, 2], label='layer3')
# plt.plot(m19base1.T[:, 3], label='layer4')
# plt.plot(m19base1.T[:, 4], label='layer5')
# plt.title('w2v2',size=14)

# plt.ylabel('abx-error', size=18)

# plt.xticks([0,1,2,3,4],['1', '5', '10', '15', '20'])
# plt.grid()
# plt.legend(fontsize=14)

# plt.subplot(2, 2, 3)
# plt.plot(m19base3.T[:, 0], label='layer1')
# plt.plot(m19base3.T[:, 1], label='layer2')
# plt.plot(m19base3.T[:, 2], label='layer3')
# plt.plot(m19base3.T[:, 3], label='layer4')
# plt.plot(m19base3.T[:, 4], label='layer5')
# plt.title('VGS+', size=14)
# plt.xlabel('epoch', size=14)
# plt.ylabel('abx-error', size=18)
# plt.xticks([0,1,2,3,4],['1', '5', '10', '15', '20'])
# plt.grid()
# plt.legend(fontsize=14)
# plt.savefig(os.path.join(path_save, 'fig2_light19_abx' + '.png'), format='png')

##################################################################
                        ### Model 20 ###
################################################################## reading from csv files
layer_names = ['L1','L2','L3','L4','L5']

# for epoch in range(5,30,5):
#     print(epoch)
#     for layer_name in layer_names:
#         name = 'E' + str(epoch) + layer_name
#         print(name) # name = 'E10L3'
#         path = os.path.join(path_input,  name , 'score_phonetic.csv')
#         name = 'E' + str(epoch) + layer_name
#         s = read_score (path)
#         scores_m20base1.append(s)

scores_m20base1 = []
for epoch in ['E1','E5','E10','E15','E20']:
    print(epoch)
    for layer_name in layer_names:
        name = 'E' + str(epoch) + layer_name
        path = os.path.join(path_input , 'm20' , 'm20base1' , 'm20base1' + epoch + layer_name ,'score_phonetic.csv')
        print(path)
        s = read_score (path)
        scores_m20base1.append(s)
m20base1 =   (np.reshape(scores_m20base1 , [5, 5])).T

scores_m20base2 = []
for epoch in ['E1','E5','E10','E15','E20']:
    print(epoch)
    for layer_name in layer_names:
        name = 'E' + str(epoch) + layer_name
        path = os.path.join(path_input , 'm20' , 'm20base2' , 'm20base2' + epoch + layer_name ,'score_phonetic.csv')
        print(path)
        s = read_score (path)
        scores_m20base2.append(s)
m20base2 =   ( np.reshape(scores_m20base2 , [5, 5]) ).T

scores_m20base3 = []
for epoch in ['E1','E5','E10','E15','E20']:
    print(epoch)
    for layer_name in layer_names:
        name = 'E' + str(epoch) + layer_name
        path = os.path.join(path_input , 'm20' , 'm20base3' , 'm20base3' + epoch + layer_name ,'score_phonetic.csv')
        print(path)
        s = read_score (path)
        scores_m20base3.append(s)
m20base3 =  ( np.reshape(scores_m20base3 , [5, 5])).T

# scores_m20base4 = []
# for epoch in ['E1','E5','E10','E15','E20']:
#     print(epoch)
#     for layer_name in layer_names:
#         name = 'E' + str(epoch) + layer_name
#         path = os.path.join(path_input , 'm20' , 'm20base4' , 'm20base4' + epoch + layer_name ,'score_phonetic.csv')
#         print(path)
#         s = read_score (path)
#         scores_m20base4.append(s)
# m20base4 =   np.reshape(scores_m20base4 , [5, 5])
kh
################################################################ Plotting

title = 'ABX-error for the light models ( grad-mul = 1) '
fig = plt.figure(figsize=(15, 10))
fig.suptitle(title, fontsize=20)

plt.subplot(2, 2, 1)  
plt.plot(m20base1[0], label='layer1')
plt.plot(m20base1[1], label='layer2')
plt.plot(m20base1[2], label='layer3')
plt.plot(m20base1[3], label='layer4')
plt.plot(m20base1[4], label='layer5')
plt.title('w2v2',size=14)  
plt.ylabel('abx-error', size=18) 
plt.xticks([0,1,2,3,4],['1', '5', '10', '15', '20'])
plt.grid()
plt.legend(fontsize=14)     

plt.subplot(2, 2, 2)  
plt.plot(m20base2[0], label='layer1')
plt.plot(m20base2[1], label='layer2')
plt.plot(m20base2[2], label='layer3')
plt.plot(m20base2[3], label='layer4')
plt.plot(m20base2[4], label='layer5')
plt.title('VGS',size=14)  
plt.xticks([0,1,2,3,4],['1', '5', '10', '15', '20'])
plt.grid()
plt.legend(fontsize=14)     


plt.subplot(2, 2, 3)  
plt.plot(m20base3[0], label='layer1')
plt.plot(m20base3[1], label='layer2')
plt.plot(m20base3[2], label='layer3')
plt.plot(m20base3[3], label='layer4')
plt.plot(m20base3[4], label='layer5')
plt.title('VGS+',size=14)  
plt.ylabel('abx-error', size=18) 
plt.xlabel('epoch', size=14)
plt.xticks([0,1,2,3,4],['1', '5', '10', '15', '20'])
plt.grid()
plt.legend(fontsize=14)     


# plt.subplot(2, 2, 1)  
# plt.plot(m20base4.T[:, 0], label='layer1')
# plt.plot(m20base4.T[:, 1], label='layer2')
# plt.plot(m20base4.T[:, 2], label='layer3')
# plt.plot(m20base4.T[:, 3], label='layer4')
# plt.plot(m20base4.T[:, 4], label='layer5')
# plt.title('VGS+ pre',size=14)  
# plt.ylabel('abx-error', size=18) 
# plt.xlabel('epoch', size=14)
# plt.xticks([0,1,2,3,4],['1', '5', '10', '15', '20'])
# plt.grid()
#plt.legend(fontsize=14)     

plt.savefig(os.path.join(path_save, 'abx_20' + '.png'), format='png')



##################################################################
                        ### Model 19 ###
################################################################## reading from csv files
layer_names = ['L1','L2','L3','L4','L5']

scores_m19base1 = []
for epoch in ['E1','E5','E10','E15','E20']:
    print(epoch)
    for layer_name in layer_names:
        name = 'E' + str(epoch) + layer_name
        path = os.path.join(path_input , 'm19' , 'm19base1' , 'm19base1' + epoch + layer_name ,'score_phonetic.csv')
        print(path)
        s = read_score (path)
        scores_m19base1.append(s)
m19base1 =   (np.reshape(scores_m19base1 , [5, 5]) ).T  

scores_m19base2 = []
for epoch in ['E1','E5','E10','E15','E20']:
    print(epoch)
    for layer_name in layer_names:
        name = 'E' + str(epoch) + layer_name
        path = os.path.join(path_input , 'm19' , 'm19base2' , 'm19base2' + epoch + layer_name ,'score_phonetic.csv')
        print(path)
        s = read_score (path)
        scores_m19base2.append(s)
m19base2 =   (np.reshape(scores_m19base2 , [5, 5])).T

scores_m19base3 = []
for epoch in ['E1','E5','E10','E15','E20']:
    print(epoch)
    for layer_name in layer_names:
        name = 'E' + str(epoch) + layer_name
        path = os.path.join(path_input , 'm19' , 'm19base3' , 'm19base3' + epoch + layer_name ,'score_phonetic.csv')
        print(path)
        s = read_score (path)
        scores_m19base3.append(s)
m19base3 =  ( np.reshape(scores_m19base3 , [5, 5])).T

scores_m19base4 = []
for epoch in ['E1','E5','E10','E15','E20']:
    print(epoch)
    for layer_name in layer_names:
        name = 'E' + str(epoch) + layer_name
        path = os.path.join(path_input , 'm19' , 'm19base4' , 'm19base4' + epoch + layer_name ,'score_phonetic.csv')
        print(path)
        s = read_score (path)
        scores_m19base4.append(s)
m19base4 =   (np.reshape(scores_m19base4 , [5, 5])).T

################################################################ Plotting

title = 'ABX-error for the light models ( grad-mul = 0.1) '
fig = plt.figure(figsize=(15, 10))
fig.suptitle(title, fontsize=20)

plt.subplot(2, 2, 1)  
plt.plot(m19base1[0], label='layer1')
plt.plot(m19base1[1], label='layer2')
plt.plot(m19base1[2], label='layer3')
plt.plot(m19base1[3], label='layer4')
plt.plot(m19base1[4], label='layer5')
plt.title('w2v2',size=14)  
plt.ylabel('abx-error', size=18) 
plt.xticks([0,1,2,3,4],['1', '5', '10', '15', '20'])
plt.grid()
plt.legend(fontsize=14)     

plt.subplot(2, 2, 2)  
plt.plot(m19base2[0], label='layer1')
plt.plot(m19base2[1], label='layer2')
plt.plot(m19base2[2], label='layer3')
plt.plot(m19base2[3], label='layer4')
plt.plot(m19base2[4], label='layer5')
plt.title('VGS',size=14)  
plt.xticks([0,1,2,3,4],['1', '5', '10', '15', '20'])
plt.grid()
plt.legend(fontsize=14)     


plt.subplot(2, 2, 3)  
plt.plot(m19base3[0], label='layer1')
plt.plot(m19base3[1], label='layer2')
plt.plot(m19base3[2], label='layer3')
plt.plot(m19base3[3], label='layer4')
plt.plot(m19base3[4], label='layer5')
plt.title('VGS+',size=14)  
plt.ylabel('abx-error', size=18) 
plt.xlabel('epoch', size=14)
plt.xticks([0,1,2,3,4],['1', '5', '10', '15', '20'])
plt.grid()
plt.legend(fontsize=14)     


plt.subplot(2, 2, 4)  
plt.plot(m19base4[0], label='layer1')
plt.plot(m19base4[1], label='layer2')
plt.plot(m19base4[2], label='layer3')
plt.plot(m19base4[3], label='layer4')
plt.plot(m19base4[4], label='layer5')
plt.title('VGS+ pre',size=14)  
plt.ylabel('abx-error', size=18) 
plt.xlabel('epoch', size=14)
plt.xticks([0,1,2,3,4],['1', '5', '10', '15', '20'])
plt.grid()
plt.legend(fontsize=14)     

plt.savefig(os.path.join(path_save, 'abx_19' + '.png'), format='png')

################################################################ 19 vs 20

title = 'ABX-error comparison for grad-mul = 0.1 and grad-mul = 1 '
fig = plt.figure(figsize=(15, 10))
fig.suptitle(title, fontsize=20)

plt.subplot(1, 2, 1)  
plt.plot(m19base1[0], label='gradmul = 0.1')
plt.plot(m20base1[0], label='gradmul = 1')
plt.title('w2v2, layer 1 (best layer)',size=14)  
plt.ylabel('abx-error', size=18) 
plt.xticks([0,1,2,3,4],['1', '5', '10', '15', '20'])
plt.grid()
plt.legend(fontsize=14)     

plt.subplot(1, 2, 2)  
plt.plot(m19base2[2], label='gradmul = 0.1')
plt.plot(m20base2[2], label='gradmul = 1')
plt.title('VGS, layer 3 (best layer)',size=14)  
plt.ylabel('abx-error', size=18) 
plt.xticks([0,1,2,3,4],['1', '5', '10', '15', '20'])
plt.grid()
plt.legend(fontsize=14)  
plt.savefig(os.path.join(path_save, 'abx_19vs20' + '.png'), format='png')    