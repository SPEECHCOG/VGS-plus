
import numpy
import os
from matplotlib import pyplot as plt

path_save = '/worktmp2/hxkhkh/current/FaST/experiments/versions/'
N = 24

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
plt.xlabel('epochs')
plt.ylabel('alpha  \n')
plt.ylim(0,1)
plt.xlim(-1,26)
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
plt.xlabel('epochs')
plt.ylabel('alpha  \n')
plt.ylim(0,1)
plt.xlim(-1,26)
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
plt.xlabel('epochs')
plt.ylabel('alpha  \n')
plt.ylim(0,1)
plt.savefig(os.path.join(path_save, name + '.png'), format='png')

# model 19T4
x = numpy.arange(0, N, 1)
y = 0.1 * numpy.ones([N])
y [8:16] = 0.5
plt.plot(x, y)
plt.grid()
name = 'model19T4'
plt.title(name)
plt.xlabel('epochs')
plt.ylabel('alpha  \n')
plt.ylim(0,1)
plt.savefig(os.path.join(path_save, name + '.png'), format='png')

# model 19T5

x = numpy.arange(0,N, 1)
m = 0.8 / 24
y = 0.1 + m * x
plt.plot(x, y)
plt.grid()
name = 'model19T5'
plt.title(name)
plt.xlabel('epochs')
plt.ylabel('alpha  \n')
plt.ylim(0,1)
plt.savefig(os.path.join(path_save, name + '.png'), format='png')

# model 19T6

x = numpy.arange(0,N, 1)
m = 0.8 / 24
y = 0.9 - m * x
plt.plot(x, y)
plt.grid()
name = 'model19T6'
plt.title(name)
plt.xlabel('epochs')
plt.ylabel('alpha  \n')
plt.ylim(0,1)
plt.savefig(os.path.join(path_save, name + '.png'), format='png')