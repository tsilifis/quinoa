import numpy as np
import kernel_py as kp 
import sys
sys.path.insert(0, '../')
from L_operator_theano import *
import matplotlib.pyplot as plt 

LL = np.load('LL.npy')
LB = np.load('LB.npy')
BB = np.load('BB.npy')


oper = HeatL()
oper.kernel = kp.RBF(1)

X = np.linspace(0., 3., 61)
X_b = np.array([0., 3.])
Y = np.zeros(42)
#c_x = np.zeros((61, 42))
x_data = np.linspace(0., 3., 42)[1:41]
Y[:40] = np.exp(-(x_data - 2.)**2)
#for i in range(61):
#	for j in range(40):
#		c_x[i,j] = oper.L_eval(X[i], x_data[j])
#	c_x[i,40] = oper.B_eval(X[i], 0.)
#	c_x[i,41] = oper.B_eval(X[i], 3.)
#	print i 

#np.save('c_x.npy', c_x)
C = np.vstack([np.hstack([LL, LB]), np.hstack([LB.T, BB]) ]) + np.diag((3. + 1e-8) * np.ones(Y.shape[0]))
print C.shape

#plt.contourf(C, 20)
#plt.show()

c_x = np.load('c_x.npy')
cov = oper.kernel.cov(X.reshape((61,1)))
L = np.linalg.cholesky(C)
scaled_data = np.linalg.solve(L.T, np.linalg.solve(L, Y))



