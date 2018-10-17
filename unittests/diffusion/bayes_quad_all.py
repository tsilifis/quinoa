import numpy as np 
import kernel_py as kp
import sys
sys.path.insert(0, '../')
from GP_integral import *
import chaos_toolbox as ct

X = np.load('U_X.npy')
Y = np.load('U_Y.npy')

Q = 20

var = np.load('U_sig_smooth_300.npy')
ell = np.load('U_ell_smooth_300.npy')

[x_quad, w] = ct.util.QuadratureRule('CC').get_rule(2, 6)
y_quad = np.load('data_concentrations_CC_quadrature_lev6.npy')[:,-1]
coeffs = ct.chaos.PolyChaos().comp_coeffs(x_quad, y_quad, w, Q, 'L')

all_m = np.zeros((30, coeffs.shape[0]))
all_C = np.zeros((30, coeffs.shape[0]))

for i in range(30):
	X_i = X[:(10*i+4),:].copy()
	Y_i = Y[:(10*i+4),:].copy()
	
	rbf = kp.RBF(X.shape[1], var[10*i], ell[10*i])

	I = IntegralPosterior(X_i, Y_i, Q, rbf)
	[m, C] = I.predict('L')
	all_m[i,:] = m.copy()
	#all_C[i,:] = C.copy()
	print i

np.save('results/U_all_m.npy', all_m)
#np.save('results/U_all_C.npy', all_C)



#import chaos_toolbox as ct 
#[quad_x, w] = ct.util.QuadratureRule().get_rule(1, 3)
#print quad_x.shape
#sin_quad = np.sin(quad_x).flatten()


import matplotlib.pyplot as plt 


plt.plot(range(m.shape[0]), m, '-x')
plt.plot(range(m.shape[0]), coeffs, '-o')
plt.fill_between(range(m.shape[0]), m-2*np.sqrt(C), m+2*np.sqrt(C), alpha = 0.4)
#plt.plot(range(Q), coeffs, '.')
plt.show()

#plt.plot(xi[:,0], q_bayes, 'x')
#plt.plot(xi[:,0], sinxi, '^', alpha = 0.4)
#plt.plot(xi[:,0], q_quad, '*')
#plt.show()