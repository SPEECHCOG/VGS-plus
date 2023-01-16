
import os
from scipy.io import loadmat
path_save = '/worktmp2/hxkhkh/current/FaST/experiments/plots/cogsci/'
import matplotlib.pyplot as plt
c_1 = 'blue'
c_2 = 'grey'
c_3 = 'green'
c_4 = 'red'
#################################################################### Recall
file_name =  "recall-average.mat"   
score = loadmat(os.path.join(path_save,file_name ))

x_recall_m7base3 = score['VGSplus05']['x'][0][0][0]
y_recall_m7base3 = score['VGSplus05']['y'][0][0][0]
r_b3 = score['VGSplus05']['norm'][0][0][0]

x_recall_m7ver4 = score['VGSpluspre']['x'][0][0][0]
y_recall_m7ver4 = score['VGSpluspre']['y'][0][0][0]
r_v4 = score['VGSpluspre']['norm'][0][0][0]

x_recall_m7base4 = score['VGSplus01']['x'][0][0][0]
y_recall_m7base4 = score['VGSplus01']['y'][0][0][0]
r_b4 = score['VGSplus01']['norm'][0][0][0]

x_recall_m7base5 = score['VGSplus09']['x'][0][0][0]
y_recall_m7base5 = score['VGSplus09']['y'][0][0][0]
r_b5 = score['VGSplus09']['norm'][0][0][0]


#################################################################### ABX
file_name =  "abx.mat"   
score = loadmat(os.path.join(path_save,file_name ))

x_abx_m7base3 = score['VGSplus05']['x'][0][0][0]
y_abx_m7base3 = score['VGSplus05']['y'][0][0][0]
abx_b3 = score['VGSplus05']['norm'][0][0][0]

x_abx_m7ver4 = score['VGSpluspre']['x'][0][0][0]
y_abx_m7ver4 = score['VGSpluspre']['y'][0][0][0]
abx_v4 = score['VGSpluspre']['norm'][0][0][0]

x_abx_m7base4 = score['VGSplus01']['x'][0][0][0]
y_abx_m7base4 = score['VGSplus01']['y'][0][0][0]
abx_b4 = score['VGSplus01']['norm'][0][0][0]

x_abx_m7base5 = score['VGSplus09']['x'][0][0][0]
y_abx_m7base5 = score['VGSplus09']['y'][0][0][0]
abx_b5 = score['VGSplus09']['norm'][0][0][0]

#################################################################### lex
file_name =  "lex.mat"   
score = loadmat(os.path.join(path_save,file_name ))

x_lex_m7base3 = score['VGSplus05']['x'][0][0][0]
y_lex_m7base3 = score['VGSplus05']['y'][0][0][0]
lex_b3 = score['VGSplus05']['norm'][0][0][0]

x_lex_m7ver4 = score['VGSpluspre']['x'][0][0][0]
y_lex_m7ver4 = score['VGSpluspre']['y'][0][0][0]
lex_v4 = score['VGSpluspre']['norm'][0][0][0]

x_lex_m7base4 = score['VGSplus01']['x'][0][0][0]
y_lex_m7base4 = score['VGSplus01']['y'][0][0][0]
lex_b4 = score['VGSplus01']['norm'][0][0][0]

x_lex_m7base5 = score['VGSplus09']['x'][0][0][0]
y_lex_m7base5 = score['VGSplus09']['y'][0][0][0]
lex_b5 = score['VGSplus09']['norm'][0][0][0]

####################################################################
#################################################################### plotting
####################################################################

fig = plt.figure(figsize=(8,20))

title = 'VGS+, alpha = 0.5'
ax = fig.add_subplot(4, 1, 1)
x_recall_m7base3 [0] = 0.5
x_abx_m7base3 [0] = 0.5
x_lex_m7base3[0] = 0.5
plt.plot(x_recall_m7base3, r_b3, c_1, label='recall@10')
plt.plot(x_abx_m7base3, abx_b3,c_2, label='abx')
plt.plot(x_lex_m7base3, lex_b3,c_3, label='lexical')
ax.set_xscale('log')
plt.xticks([0.5,1,2,3,4,5,10,50],['0','1','2','3','4','5','10','50'])
plt.ylim(0,1)
plt.grid()
plt.legend(fontsize=12)
plt.title(title)


title = 'VGS+, Pre'
ax = fig.add_subplot(4, 1, 2)
x_recall_m7base4 [0] = 0.5
x_abx_m7base4 [0] = 0.5
x_lex_m7base4[0] = 0.5
plt.plot(x_recall_m7ver4, r_v4, c_1, label='recall@10')
plt.plot(x_abx_m7ver4, abx_v4,c_2, label='abx')
plt.plot(x_lex_m7ver4, lex_v4,c_3, label='lexical')
ax.set_xscale('log')
plt.xticks([0.5,1,2,3,4,5,10,50],['0','1','2','3','4','5','10','50'])
plt.ylim(0,1)
plt.grid()
plt.legend(fontsize=12)
plt.title(title)


title = 'VGS+, alpha = 0.1'
ax = fig.add_subplot(4, 1, 3)
x_recall_m7base4 [0] = 0.5
x_abx_m7base4 [0] = 0.5
x_lex_m7base4[0] = 0.5
plt.plot(x_recall_m7base4, r_b4, c_1, label='recall@10')
plt.plot(x_abx_m7base4, abx_b4,c_2, label='abx')
plt.plot(x_lex_m7base4, lex_b4,c_3, label='lexical')
ax.set_xscale('log')
plt.xticks([0.5,1,2,3,4,5,10,50],['0','1','2','3','4','5','10','50'])
plt.ylim(0,1)

plt.grid()
plt.legend(fontsize=12)
plt.title(title)


title = 'VGS+, alpha = 0.9 '
ax = fig.add_subplot(4, 1, 4)
x_recall_m7base5 [0] = 0.5
x_abx_m7base5 [0] = 0.5
x_lex_m7base5[0] = 0.5
plt.plot(x_recall_m7base5, r_b5, c_1, label='recall@10')
plt.plot(x_abx_m7base5, abx_b5,c_2, label='abx')
plt.plot(x_lex_m7base5, lex_b5,c_3, label='lexical')
ax.set_xscale('log')
plt.xticks([0.5,1,2,3,4,5,10,50],['0','1','2','3','4','5','10','50'])
plt.ylim(0,1)
plt.grid()
plt.legend(fontsize=12)
plt.title(title)
plt.xlabel('Epoch',size=18)

plt.savefig(os.path.join(path_save, 'all_normalized_logx' + '.png'), format='png')