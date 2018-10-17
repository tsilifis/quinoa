import numpy as np 
import kernel_py as kp
import sys
sys.path.insert(0, '../')
from GP_integral import *
import chaos_toolbox as ct

X = np.load('UU_X.npy')[:290,:]
Y = np.load('UU_Y.npy')[:290,:]

Q = 5

pi = np.zeros((X.shape[0],Q))
rho = np.zeros(Q)

var = np.load('UU_sig.npy')[290]
ell = np.load('UU_ell.npy')[290]


rbf = kp.RBF(X.shape[1], var, ell)

I = IntegralPosterior(X, Y, Q, rbf)
[m, C] = I.predict('L')
#print m

#print C
[x_quad, w] = ct.util.QuadratureRule('CC').get_rule(5, 6)
#print x_quad.shape

#[x_quad, w] = ct.util.QuadratureRule('CC').get_rule(2, 8)
y_quad = np.load('data_pressure_CC_quadrature_lev6.npy')[:,-1]
coeffs = ct.chaos.PolyChaos().comp_coeffs(x_quad, y_quad, w, Q, 'L')

#for i in range(Q):
#	for j in range(X.shape[0]):
#		pi[j,i] = I.pi(i, X[j,0], ell)
#	rho[i] = I.rho(i,i, ell)



#v = var * pi.T / np.sqrt(2 * np.pi)
#cov = rbf.cov(X)
#L = np.linalg.cholesky(cov)
#scaled_data = np.linalg.solve(L.T, np.linalg.solve(L, Y[:,0])).flatten()
#m = np.dot(v, scaled_data)
#T = np.linalg.solve(L, v.T)
#print T.shape
#C1 = var * rho / (2*np.pi)
#print np.dot(T.T, T).shape
#C = C1 - np.diag(np.dot(T.T, T))

#print C


#import chaos_toolbox as ct 
#[quad_x, w] = ct.util.QuadratureRule().get_rule(1, 3)
#print quad_x.shape
#sin_quad = np.sin(quad_x).flatten()


#coeffs = ct.chaos.PolyChaos().comp_coeffs(quad_x, sin_quad, w, Q-1)


#pol = ct.chaos.PolyBasis(1, Q-1)
#xi = np.linspace(-4., 4., 100).reshape(100,1)
#P = pol(xi)
#sinxi = np.sin(xi)
#q_bayes = np.dot(P, m)
#q_quad = np.dot(P, coeffs)

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