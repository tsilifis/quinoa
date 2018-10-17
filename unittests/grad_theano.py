import numpy as np 
import kernel_py as kp
import theano 
import theano.tensor as T 
from theano import pp

x = T.dscalar('x')
y = kp.RBF(1).eval(x,0.)
gy = T.grad(y, x)
pp(gy)
f = theano.function([x], gy)
print f(2)
vals = np.linspace(-3., 3, 50)
ker_grad = np.zeros(vals.shape)
for i in range(vals.shape[0]):
	ker_grad[i] = f(vals[i])


import matplotlib.pyplot as plt 
print np.array([[0.]]).shape
grad_n = kp.RBF(1).d_cov_d_X(vals.reshape(50, 1), np.array([[0.]]))
print grad_n.shape

plt.plot(vals, ker_grad, '-')
plt.plot(vals, grad_n[0,:,0], '-.')
plt.show()