
import numpy
import os
from matplotlib import pyplot as plt

path_save = '/worktmp2/hxkhkh/current/FaST/experiments/versions/'
N = 24

kh

# model 19T1
T = 2 * N
x = numpy.arange(0,N,1)
a = (2*numpy.pi) / T
y0 = 0.1
m = 0.8
y = y0 + m * ((numpy.sin(a*x))**2)
plt.plot(x,y);plt.grid()
name = 'model19T1'
plt.title(name)
plt.xlabel('n_steps for 24 epochs')
plt.ylabel('alpha  \n')
plt.ylim(0,1)
plt.savefig(os.path.join(path_save , name + '.png'), format = 'png')

# model 19T2
T = 1 * N
x = numpy.arange(0,N,1)
a = (2*numpy.pi) / T
y0 = 0.1
m = 0.8
y = y0 + m * ((numpy.sin(a*x))**2)
plt.plot(x,y);plt.grid()
name = 'model19T2'
plt.title(name)
plt.xlabel('n_steps for 24 epochs')
plt.ylabel('alpha  \n')
plt.ylim(0,1)
plt.savefig(os.path.join(path_save , name + '.png'), format = 'png')

# model 19T3
T = 2 * N
x = numpy.arange(0, N, 1)
a = (2*numpy.pi) / T
y0 = 0.1
m = 0.4
y = y0 + m * ((numpy.sin(a*x))**2)
plt.plot(x, y)
plt.grid()
name = 'model19T3'
plt.title(name)
plt.xlabel('n_steps for 24 epochs')
plt.ylabel('alpha  \n')
plt.ylim(0,1)
plt.savefig(os.path.join(path_save, name + '.png'), format='png')

# model 19T4