import numpy as np 
from GP_integral import *
import kernel_py as kp


X = np.load('X.npy')[:4,:]
Y = np.load('Y.npy')[:4,:]
Q = 9

I = IntegralPosterior(X, Y, Q)

pi = np.zeros((X.shape[0],Q))
rho = np.zeros(Q)

var = np.load('sig.npy')[3]
ell = np.load('ell.npy')[3]

rbf = kp.RBF(1, var, ell)

for i in range(Q):
	for j in range(X.shape[0]):
		pi[j,i] = I.pi(i, X[j,0], ell)
	rho[i] = I.rho(i,i, ell)



v = var * pi.T / np.sqrt(2 * np.pi)
cov = rbf.cov(X)
L = np.linalg.cholesky(cov)
scaled_data = np.linalg.solve(L.T, np.linalg.solve(L, Y[:,0])).flatten()
m = np.dot(v, scaled_data)
T = np.linalg.solve(L, v.T)
print T.shape
C1 = var * rho / (2*np.pi)
#print np.dot(T.T, T).shape
C = C1 - np.diag(np.dot(T.T, T))

print C


import chaos_toolbox as ct 
[quad_x, w] = ct.util.QuadratureRule().get_rule(1, 3)
print quad_x.shape
sin_quad = np.sin(quad_x).flatten()


coeffs = ct.chaos.PolyChaos().comp_coeffs(quad_x, sin_quad, w, Q-1)
#print coeffs.shape


pol = ct.chaos.PolyBasis(1, Q-1)
xi = np.linspace(-4., 4., 100).reshape(100,1)
P = pol(xi)
sinxi = np.sin(xi)
q_bayes = np.dot(P, m)
q_quad = np.dot(P, coeffs)

import matplotlib.pyplot as plt 

#plt.plot(range(Q), m, '-x')

#plt.fill_between(range(Q), m-2*np.sqrt(np.diag(C1)), m+2*np.sqrt(np.diag(C1)), alpha = 0.4)

plt.plot(range(Q), m, '-x')
plt.fill_between(range(Q), m-2*np.sqrt(C), m+2*np.sqrt(C), alpha = 0.4)
plt.plot(range(Q), coeffs, '.')
plt.show()

plt.plot(xi[:,0], q_bayes, 'x')
plt.plot(xi[:,0], sinxi, '^', alpha = 0.4)
plt.plot(xi[:,0], q_quad, '*')
plt.show()