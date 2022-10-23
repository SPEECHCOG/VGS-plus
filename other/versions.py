
import numpy
import os
from matplotlib import pyplot as plt

path_save = '/worktmp2/hxkhkh/current/FaST/experiments/versions/'
N = 24
x = numpy.arange(0,N,1)
y_lr = 0.0001 - (0.0001/N) * x
plt.plot(x,y_lr);plt.grid()
kh
# model 19T1
T = 2 * N
x = numpy.arange(0,N,1)
a = (2*numpy.pi) / T
y0 = 0.1
m = 0.8
y = y0 + m * ((numpy.sin(a*x))**2)
z = 1- y
plt.plot(x,y*y_lr, z*y_lr);plt.grid()
name = 'sin (A = 1 , f =1)'
plt.title(name)
plt.xlabel('epochs')
plt.ylabel('alpha  \n')
#plt.ylim(0,1)
plt.xlim(-1,26)
plt.savefig(os.path.join(path_save , name + '_lr.png'), format = 'png')

# model 19T2
T = 1 * N
x = numpy.arange(0,N,1)
a = (2*numpy.pi) / T
y0 = 0.1
m = 0.8
y = y0 + m * ((numpy.sin(a*x))**2)
z = 1- y
plt.plot(x,y*y_lr, z*y_lr);plt.grid()
name = 'sin (A = 1 , f =2)'
plt.title(name)
plt.xlabel('epochs')
plt.ylabel('alpha  \n')
#plt.ylim(0,1)
plt.xlim(-1,26)
plt.savefig(os.path.join(path_save , name + '_lr.png'), format = 'png')

# model 19T3
T = 2 * N
x = numpy.arange(0, N, 1)
a = (2*numpy.pi) / T
y0 = 0.1
m = 0.4
y = y0 + m * ((numpy.sin(a*x))**2)
z = 1- y
plt.plot(x, y*y_lr, z*y_lr)
plt.grid()
name = 'sin (A = 0.5)'
plt.title(name)
plt.xlabel('epochs')
plt.ylabel('alpha  \n')
#plt.ylim(0,1)
plt.savefig(os.path.join(path_save, name + '_lr.png'), format='png')

# model 19T4
x = numpy.arange(0, N, 1)
y = 0.1 * numpy.ones([N])
y [8:16] = 0.5
z = 1- y
plt.plot(x, y*y_lr, z*y_lr)
plt.grid()
name = 'step function'
plt.title(name)
plt.xlabel('epochs')
plt.ylabel('alpha  \n')
#plt.ylim(0,1)
plt.savefig(os.path.join(path_save, name + '_lr.png'), format='png')

# model 19T5

x = numpy.arange(0,N, 1)
m = 0.8 / 24
y = 0.1 + m * x
z = 1- y
plt.plot(x, y*y_lr, z*y_lr)
plt.grid()
name = 'linear increasing'
plt.title(name)
plt.xlabel('epochs')
plt.ylabel('alpha  \n')
#plt.ylim(0,1)
plt.savefig(os.path.join(path_save, name + '_lr.png'), format='png')

# model 19T6

x = numpy.arange(0,N, 1)
m = 0.8 / 24
y = 0.9 - m * x
z = 1- y
plt.plot(x, y*y_lr, z*y_lr)
plt.grid()
name = 'linear decreasing'
plt.title(name)
plt.xlabel('epochs')
plt.ylabel('alpha  \n')
#plt.ylim(0,1)
plt.savefig(os.path.join(path_save, name + '_lr.png'), format='png')

# model 19T7
x = numpy.arange(0,N, 1)
y = 0.1 * numpy.ones(24)
z = 1- y
plt.plot(x, y*y_lr, z*y_lr)
plt.grid()
name = 'alpha = 0.1'
plt.title(name)
plt.xlabel('epochs')
plt.ylabel('alpha  \n')
#plt.ylim(0,1)
plt.savefig(os.path.join(path_save, name + '_lr.png'), format='png')


# model 19T8
x = numpy.arange(0,N, 1)
y = 0.9 * numpy.ones(24)
z = 1- y
plt.plot(x, y*y_lr, z*y_lr)
plt.grid()
name = 'alpha = 0.9'
plt.title(name)
plt.xlabel('epochs')
plt.ylabel('alpha  \n')
#plt.ylim(0,1)
plt.savefig(os.path.join(path_save, name + '_lr.png'), format='png')


# model 19T9
x = numpy.arange(0,N, 1)
y = 0.9 * numpy.ones(24)
y[::2]= 0.1
z = 1- y
plt.scatter(x, y*y_lr, z*y_lr)
plt.plot(x,y, '--')
plt.grid()
name = 'model19T9'
plt.title(name)
plt.xlabel('epochs')
plt.ylabel('alpha  \n')
#plt.ylim(0,1)
plt.savefig(os.path.join(path_save, name + '_lr.png'), format='png')